from flask import Flask, render_template, redirect, url_for, flash, session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, ValidationError
import bcrypt

# Flask app setup
app = Flask(__name__)

app.secret_key = 'your_secret_key_here'

# Registration form setup using Flask-WTF
class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Register")

    # Custom email validation to check if email is already taken
    def validate_email(self, field):
        # This is an example; replace this with actual DB email check logic
        if field.data == 'test@example.com':  
            raise ValidationError('Email is already taken')

# Home route (index)
@app.route('/')
def index():
    total_spots = 50  # Example value
    available_spots = 20  # Example value
    return render_template('index.html', total_spots=total_spots, available_spots=available_spots)

# Register route (renders the register form)
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data

        # Hash the password before storing it (example with bcrypt)
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Save to your database here (example: store the user info)
        # Example: cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, hashed_password))

        # Flash the success message after successful registration
        flash("Registration successful! Welcome to the system.", "success")

        # Redirect to home page after successful registration
        return redirect(url_for('index'))  

    return render_template('register.html', form=form)

# Login route (renders the login form)
@app.route('/login', methods=['GET', 'POST'])
def login():
    # You would have your login form here, for simplicity, we skip the details.
    flash("Logged in successfully", "success")
    return redirect(url_for('index'))  # Redirect to home page

# Logout route (logs the user out)
@app.route('/logout')
def logout():
    session.clear()  # Clears the session to log out
    flash("Logged out successfully", "success")
    return redirect(url_for('index'))  # Redirect to home page

if __name__ == '_main_':
    app.run(debug=True)