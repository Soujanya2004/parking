from flask import Flask, render_template, Response, request, redirect, url_for, session, flash, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, ValidationError
from flask_mysqldb import MySQL
import bcrypt
import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not

# ----------------------
# App Initialization
# ----------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ----------------------
# MySQL Configuration
# ----------------------
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'soujusonu@20'
app.config['MYSQL_DB'] = 'mydatabase'

mysql = MySQL(app)

# ----------------------
# Authentication Forms
# ----------------------
class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Register")

    def validate_email(self, field):
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s", (field.data,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            raise ValidationError('Email Already Taken')

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

# ----------------------
# Parking Detection Config
# ----------------------
mask_path = r'D:\model-parking\parking-space-counter\mask_1920_1080.png'
video_path = r'D:\model-parking\parking-space-counter\parking_1920_1080.mp4'

mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
spot_numbers = [i for i in range(len(spots))]
spots_status = [None for _ in spots]
diffs = [None for _ in spots]
previous_frame = None
frame_nmr = 0
step = 30

# ----------------------
# Video Frame Generator
# ----------------------
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

def generate_frames():
    global previous_frame, frame_nmr, spots_status
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if end is reached
            continue

        if frame_nmr % step == 0:
            if previous_frame is not None:
                for spot_indx, spot in enumerate(spots):
                    x1, y1, w, h = spot
                    spot_crop = frame[y1:y1 + h, x1:x1 + w]
                    diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w])

                for spot_indx in range(len(spots)):
                    x1, y1, w, h = spots[spot_indx]
                    spot_crop = frame[y1:y1 + h, x1:x1 + w]
                    spots_status[spot_indx] = empty_or_not(spot_crop)

        previous_frame = frame.copy()

        for spot_indx, (x1, y1, w, h) in enumerate(spots):
            spot_status = spots_status[spot_indx]
            color = (0, 255, 0) if spot_status else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            cv2.putText(frame, str(spot_numbers[spot_indx]), (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        available_spots = sum(1 for status in spots_status if status)
        total_spots = len(spots_status)
        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        cv2.putText(frame, f'Available spots: {available_spots} / {total_spots}', (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        frame_nmr += 1

# ----------------------
# Routes
# ----------------------
@app.route('/')
def home():
    return redirect(url_for('register'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, hashed_password))
        mysql.connection.commit()
        cursor.close()

        session['user_id'] = email
        flash("Registration successful! Welcome to the system.", "success")
        return redirect(url_for('dashboard'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
            session['user_id'] = user[0]
            return redirect(url_for('dashboard'))
        else:
            flash("Login failed. Check email and password.", "danger")
    return render_template('login.html', form=form)

@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        total_spots = len(spots_status)
        available_spots = sum(1 for status in spots_status if status)

        return render_template('index.html', total_spots=total_spots, available_spots=available_spots)
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/space_count')
def space_count():
    try:
        free_spaces = sum(1 for status in spots_status if status)
        occupied_spaces = len(spots_status) - free_spaces
        return jsonify(free=free_spaces, occupied=occupied_spaces)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/get_parking_data')
def get_parking_data():
    total_spots = len(spots_status)
    available_spots = sum(1 for status in spots_status if status)
    return jsonify(total_spots=total_spots, available_spots=available_spots)

@app.route('/book', methods=['GET', 'POST'])
def book():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    if request.method == 'POST':
        try:
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT id, spot_number FROM spots WHERE is_assigned = FALSE LIMIT 1")
            spot = cursor.fetchone()

            if spot:
                spot_id, spot_number = spot
                cursor.execute("""
                    UPDATE spots
                    SET is_assigned = TRUE, assigned_to = %s
                    WHERE id = %s
                """, (user_id, spot_id))
                mysql.connection.commit()

                flash(f"Spot {spot_number} assigned successfully!", "success")
                return render_template('book.html', assigned_spot=spot_number)

            flash("No available spots at the moment. Please try again later.", "danger")
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")
        finally:
            cursor.close()

    return render_template('book.html', assigned_spot=None)

# Main Execution
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
