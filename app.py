from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify
import os
from flask import Flask, session
from flask_session import Session
import pandas as pd
#import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pdfplumber
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from flask_socketio import SocketIO
import time
#import redis
from dotenv import load_dotenv
load_dotenv()
#from google import genai
import os
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import base64
from google.auth.transport.requests import Request
import re
from urllib.parse import urlparse
import base64
import json
import time
#from urllib.parse import urlparse  # (optional, but fine to keep)
from functools import wraps
from flask import redirect, url_for
import smtplib
import traceback
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import base64



os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")

# Force filesystem sessions (no redis)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_KEY_PREFIX"] = "myapp:"
app.config["SESSION_FILE_DIR"] = "/tmp/flask_session_files"

os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)

Session(app)



socketio = SocketIO(app, cors_allowed_origins="*")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your_secret_key'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
#client = genai.Client(api_key="AIzaSyDz3IIqX2E74NaYnXK3CmnKRBOCZekOa8A")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://promo-shots.mesaki.in/oauth2callback/google")
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/gmail.send", "openid", "https://www.googleapis.com/auth/userinfo.email"]


def create_google_flow():
    """Create an OAuth Flow object for Google login."""
    return Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=GOOGLE_SCOPES,
        redirect_uri=GOOGLE_REDIRECT_URI,
    )

def require_google_login(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        creds = get_google_creds()
        if not creds or not creds.valid:
            return redirect(url_for("home"))
        return view_func(*args, **kwargs)
    return wrapper

def get_google_creds():
    data = session.get("google_creds")
    if not data:
        return None

    creds = Credentials(
        token=data["token"],
        refresh_token=data.get("refresh_token"),
        token_uri=data["token_uri"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        scopes=data["scopes"],
    )

    # Refresh token if expired
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Save the refreshed token back into the session
        session["google_creds"] = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
        }

    return creds



@app.route("/auth/google")
def google_auth():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return "Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.", 500

    flow = create_google_flow()
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="false",  # IMPORTANT: do not auto-merge old scopes
        prompt="consent",                # IMPORTANT: force re-consent so gmail scope is granted
    )
    session["oauth_state"] = state
    return redirect(authorization_url)


@app.route("/oauth2callback/google")
def google_oauth_callback():
    flow = create_google_flow()

    # Handle the callback URL
    auth_response = request.url
    if auth_response.startswith("http://"):
        auth_response = auth_response.replace("http://", "https://", 1)

    flow.fetch_token(authorization_response=auth_response)

    creds = flow.credentials

    # Save the credentials in the session
    session["google_creds"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }

    return redirect(url_for("upload_files"))

@app.route("/logout")
def logout():
    """Simple logout: clear session and send back to home."""
    session.clear()
    return redirect(url_for("home"))

def initialize_llm(llm_provider):
    if llm_provider == "deepseek":
        return ChatOpenAI(model="deepseek/deepseek-chat",openai_api_key=OPENROUTER_API_KEY,openai_api_base="https://openrouter.ai/api/v1", temperature=0)
    elif llm_provider == 'gemini':
        return ChatOpenAI(model="google/gemini-2.0-flash-001",openai_api_key=OPENROUTER_API_KEY,openai_api_base="https://openrouter.ai/api/v1", temperature=0)
    # Replace with DeepSeek's chat model
    return ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0) # Default to OpenAI


#llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Extract text from resume

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n" if page.extract_text() else ""
    return text

current_date = datetime.today().strftime('%B %d, %Y')

def fetch_website_info(website_url):
    """Fetches the main content of a company's website to personalize the cover letter."""
    if not website_url or not website_url.startswith("http"):
        return None  # Ensure the website URL is valid

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(website_url, headers=headers, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            if paragraphs:
                return paragraphs[0].get_text().strip()
            headers = soup.find_all(["h1", "h2"])
            if headers:
                return headers[0].get_text().strip()

    except requests.RequestException:
        return None

    return None



import json

@app.route('/upload', methods=['GET', 'POST'])
@require_google_login
def upload_files():
    if request.method == 'POST':
        session.pop("email_path", None)
        session.pop("resume_paths", None)
        session.pop("emails_data", None)


        if 'email_file' not in request.files or 'resume_files' not in request.files:
            return jsonify({"error": "Missing files"}), 400

        email_file = request.files['email_file']
        resume_files = request.files.getlist('resume_files')

        if email_file.filename == "" or len(resume_files) == 0:
            return jsonify({"error": "No files selected"}), 400

        email_path = os.path.join(app.config['UPLOAD_FOLDER'], email_file.filename)
        email_file.save(email_path)

        resume_texts = {}
        resume_paths = {}

        for resume_file in resume_files:
            if resume_file.filename == "":
                continue  # Ignore empty filenames

            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(resume_path)

            resume_texts[resume_file.filename] = extract_text_from_pdf(resume_path)
            resume_paths[resume_file.filename] = resume_path

        # Save resume texts to a JSON file instead of the session
        with open("resume_texts.json", "w") as f:
            json.dump(resume_texts, f)

        session['email_path'] = email_path
        session['resume_paths'] = resume_paths  # Keep only file paths in session

        return jsonify({"success": True, "message": "Files uploaded successfully!"})

    return render_template('premade_cover.html')



    

    

@app.route("/premade_cover", methods=["GET"])
def premade_cover():
    return render_template("premade_cover.html")




    

@app.route('/review', methods=['GET', 'POST'])
def review_emails():
    emails_data = session.get('emails_data')

    if not emails_data:
        return redirect(url_for('upload_files'))

    for email in emails_data:
        if 'job_description' not in email:
            email['job_description'] = "No job description available."

    return render_template('review.html', emails_data=emails_data)



 # "dclo ewei hyrg ltar"

from flask import Flask, render_template, request, redirect, url_for, session, jsonify

def send_message_via_gmail_api(creds, to_email, subject, body_text, attachment_path=None, bcc_email=None):
    """Send a single email with optional attachment using Gmail API."""
    service = build("gmail", "v1", credentials=creds)

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email['recipient_email']
    msg['Bcc'] = sender_email  # ✅ Adds sender in BCC
    msg['Subject'] = f"Job Application for {email['job_position']}"

    msg.attach(MIMEText(email['cover_letter'], 'plain'))

            # ✅ Retrieve the correct resume path
    resume_filename = email.get("selected_resume")  # The selected resume filename
    resume_path = resume_paths.get(resume_filename)  # Get actual path from stored resumes

    if not resume_path:
        print(f"❌ Resume path not found for {resume_filename}")
        return None  # Skip this email if resume is missing

    with open(resume_path, 'rb') as resume_file:
        attach_file = MIMEBase('application', 'octet-stream')
        attach_file.set_payload(resume_file.read())
        encoders.encode_base64(attach_file)
        attach_file.add_header('Content-Disposition', f'attachment; filename={os.path.basename(resume_path)}')
        msg.attach(attach_file)

    try:
        send_message = service.users().messages().send(userId="me", body=message).execute()
        print(f"Message Id: {send_message['id']}")
        return send_message
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


@app.route('/send_email', methods=['POST'])
@require_google_login
def send_email():
    creds = get_google_creds()
    if not creds:
        return jsonify({
            "success": False,
            "message": "Google account not connected. Please go back to Home and click 'Connect with Google' first."
        }), 401

    # Proceed with the rest of your logic to send emails using the Gmail API...





@app.route('/success')
def success():
    return render_template('success.html')


@app.route("/")
def home():
    creds = get_google_creds()
    google_connected = creds is not None and creds.valid
    return render_template("home.html", google_connected=google_connected)


@app.route('/view_resume/<filename>')
def view_resume(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



@app.route('/send_premade_cover', methods=['POST'])
@require_google_login
def send_premade_cover():
    sender_email = "meerahmed@gmail.com"
    sender_password = "dqqo uhqy tjbz zuok"  # This is no longer needed since we're using Gmail API
    smtp_server = "smtp.gmail.com"  # No longer needed
    smtp_port = 587  # No longer needed

    email_file = request.files.get('email_file')
    flyer_file = request.files.get('flyer_image')
    attachment_files = request.files.getlist('attachments')

    if not email_file or not flyer_file:
        return jsonify({"success": False, "message": "Email file and flyer image are required."})

    # Save flyer
    flyer_path = os.path.join(app.config['UPLOAD_FOLDER'], "flyer_image.png")
    flyer_file.save(flyer_path)

    try:
        df = pd.read_excel(email_file) if email_file.filename.endswith(".xlsx") else pd.read_csv(email_file)

        if 'Email' not in df.columns or 'NAME' not in df.columns:
            return jsonify({"success": False, "message": "Excel must contain 'Email' and 'COLLEGE NAME' columns."})

        # Get Google credentials from the session
        creds = get_google_creds()
        if not creds:
            return jsonify({"success": False, "message": "Google account not connected. Please go back to Home and click 'Connect with Google' first."}), 401

        for _, row in df.iterrows():
            recipient_email = row.get('Email')
            college_name = row.get('NAME', 'Your College')

            subject_line = f"Proposal for Strategic Partnership with {college_name}"

            html_body = f"""
            <p>Respected Dean/Principal of <b>{college_name}</b>,</p>
            <p>I hope this message finds you well.</p>
            <p>In today’s rapidly evolving job market, the need to bridge the gap between academic education and real-world industry expectations is more urgent than ever. With this in mind, I’m excited to share with you a proposal that outlines a comprehensive, future-ready roadmap to enhance student employability, technological exposure, and industry alignment for your esteemed institution.</p>
            <p>My research, attached herewith, is the result of years of experience across global tech firms such as IBM and HP, and my ongoing efforts through CognAILabs and iREAD Foundation to empower students, especially from underrepresented backgrounds, with the tools and mentorship they need to succeed.</p>
            <p>This document includes:</p>
            <ul>
                <li>A critical analysis of the current skills gap</li>
                <li>A 14-point service model for immediate implementation</li>
                <li>A partnership blueprint that builds on your institution’s strengths</li>
            </ul>
            <p>I invite you to review this proposal and explore potential avenues where we can collaborate to shape career-ready graduates. I would be honored to discuss how we can tailor this initiative to meet your students’ and faculty’s unique needs.</p>
            <p><img src="cid:flyerimage" style="max-width:100%; height:auto;" /></p>
            <p>Thank you for your time and consideration. I look forward to the opportunity to connect further.</p>
            <br>
            <p>
            Warm regards,<br>
            Meer Ahmed<br>
            Ex-IBM & HP<br>
            Founder & Director – CognAILabs<br>
            Managing Trustee – iREAD Foundation<br>
            📞 +91 9900058138<br>
            📩 meer@cognizeal.com
            </p>
            """

            msg = MIMEMultipart('related')
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject_line
            msg['Bcc'] = sender_email

            alt_part = MIMEMultipart('alternative')
            alt_part.attach(MIMEText(html_body, 'html'))
            msg.attach(alt_part)

            # Attach flyer image inline
            with open(flyer_path, 'rb') as f:
                flyer_data = f.read()
                flyer_img = MIMEImage(flyer_data)
                flyer_img.add_header('Content-ID', '<flyerimage>')
                flyer_img.add_header('Content-Disposition', 'inline', filename="flyer.png")
                msg.attach(flyer_img)

            # Attach uploaded documents (multiple)
            for file in attachment_files:
                if file and file.filename:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(filepath)
                    with open(filepath, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(filepath)}"')
                        msg.attach(part)

            # Send email using Gmail API
            send_message_via_gmail_api(creds, recipient_email, subject_line, html_body, flyer_path)

        return jsonify({"success": True, "message": "Emails sent successfully!"})

    except Exception as e:
        print("❌ Error sending emails:", str(e))
        return jsonify({"success": False, "message": f"Error: {str(e)}"})



    except Exception as e:
        print("❌ Error sending emails:", str(e))
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Error: {str(e)}"})





@app.route('/preview_cover_letter', methods=['POST'])
def preview_cover_letter():
    try:
        llm_provider = request.form.get('llm_provider', 'openai')
        custom_prompt = request.form.get('custom_prompt', '').strip()
        university_name = row.get('NAME', 'Your College')
        subject = f"Proposal for Strategic Partnership with {university_name}"


        # Try to extract base cover letter
        cover_letter_text = request.form.get('cover_letter_text', '').strip()

        if not cover_letter_text:
            uploaded_file = request.files.get('cover_letter_file')
            if uploaded_file and uploaded_file.filename:
                filename = secure_filename(uploaded_file.filename)
                ext = filename.lower().split('.')[-1]

                if ext == "txt":
                    cover_letter_text = uploaded_file.read().decode('utf-8')
                elif ext == "docx":
                    import io
                    #import docx
                    docx_bytes = cover_letter_file.read()
                    doc = docx.Document(io.BytesIO(docx_bytes))
                    cover_letter = "\n".join([para.text for para in doc.paragraphs])

                else:
                    return jsonify({"success": False, "message": "Unsupported cover letter file type."})

        if not cover_letter_text:
            return jsonify({"success": False, "message": "No cover letter content found."})

        # Fake sample values for testing preview
        sample_company = "Acme Corp"
        sample_position = "Innovation Strategist"

        # Replace basic placeholders for prompt usage
        input_to_llm = f"""
You are an assistant helping the user customize a pre-written cover letter.
Below is the user's prompt with instructions, and a sample cover letter.

Please apply the instructions strictly to generate the customized cover letter.

--- USER INSTRUCTIONS ---
{custom_prompt if custom_prompt else 'Replace XXXXX with company name and YYYYY with job position.'}

--- BASE COVER LETTER ---
{cover_letter_text}

--- VARIABLES ---
Company Name: {sample_company}
Job Position: {sample_position}
        """

        # Call LLM
        llm = initialize_llm(llm_provider)
        response = llm.invoke(input_to_llm).content.strip()

        return jsonify({"success": True, "preview": response})

    except Exception as e:
        print(f"❌ Preview error: {e}")
        return jsonify({"success": False, "message": f"Error generating preview: {str(e)}"})




if __name__ == '__main__':
    #socketio.run(app,debug=True)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

