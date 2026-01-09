import smtplib
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import getpass
import pdfplumber
from datetime import datetime

# --- CONFIGURATION ---
XL_FILE = "emails.xlsx"  # Excel file containing email addresses and job details
RESUME_PATH = "Meer_Ahmed_Resume_Sep2024_Ver1.1.pdf"
SENDER_EMAIL = "madiha1ahmed@gmail.com"  # Your email address
SENDER_PASSWORD = getpass.getpass('Please enter your password:') # Your email password

SMTP_SERVER = "smtp.gmail.com"  # Change this for other providers (Outlook/Yahoo)
SMTP_PORT = 587  # SMTP port (usually 587 for TLS)
OPENROUTER_API_KEY = "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"

# Extract text from the resume using pdfplumber (more reliable for scanned documents)
def extract_text_with_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n" if page.extract_text() else ""
    return text

# Extract resume text again using pdfplumber
resume_text = extract_text_with_pdfplumber(RESUME_PATH)

# --- FUNCTION TO SCRAPE WEBSITE INFORMATION ---
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


# --- FUNCTION TO GENERATE COVER LETTER ---
def generate_cover_letter(company_name, job_position, job_description, website_info, resume_text):
    """
    Generates a truthful and personalized cover letter email using DeepSeek model via OpenRouter.
    """

    # Ensure that website information is included properly if available
    website_section = (
        f"\nI reviewed the information available on {company_name}'s website and was particularly drawn to {website_info}. "
        if website_info and website_info.lower() != "no website available"
        else ""
    )

    # Get the current date
    current_date = datetime.today().strftime('%B %d, %Y')


    prompt_text = f"""You are a **highly precise and fact-based** job application assistant. 
    Your task is to generate a **truthful, concise, and professional** cover letter based **ONLY on the provided resume**.

    **Candidate Details:**
    - **Name:** [Name from {resume_text}]
    - **Address:** [Address from {resume_text}]
    - **Phone Number:** [Phone Number from {resume_text}]
    - **Date:** {current_date}

    **Job Application Details:**
    - **Company:** {company_name}
    - **Position:** {job_position}
    - **Job Description:** {job_description}

    **Candidate's Resume:**
    {resume_text}

    {website_section}

    **STRICT RULES:**
    - **DO NOT fabricate any experiences, degrees, or skills that are NOT present in the resume.**
    - **Only extract relevant skills and qualifications from the resume text above.**
    - **If a required skill is missing, acknowledge it politely and express eagerness to learn.**
    - **No generic statements like "passionate about teaching" unless proven from the resume.**
    - **If website insights are available, integrate them naturally.**
    - **No more than TWO paragraphs.**

    **Example Cover Letter Structure:**
    ---
    Dear Hiring Team of {company_name},

    I am excited to apply for the {job_position} role at {company_name}. My background in **[relevant expertise from {resume_text}]** has enabled me to **[explain how your experience aligns with job responsibilities]**. {website_section}

    I look forward to the opportunity to discuss how my expertise in **[relevant experience in {resume_text}]** can contribute to **[company's goal or project mentioned in job description]**.

    Sincerely,  
    [Name from {resume_text}] 
    [City, Country from Address in {resume_text}]
    Phone: [Phone number from {resume_text}]
    ---
    """

    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Initialize the ChatOpenAI model using DeepSeek via OpenRouter
    llm = ChatOpenAI(
        model="deepseek/deepseek-chat",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1"
    )

    chain = prompt | llm

    # Pass input variables to the model
    response = chain.invoke({
        "company_name": company_name,
        "job_position": job_position,
        "job_description": job_description,
        "resume_text": resume_text,  # Explicitly passing the extracted resume
        "website_info_integration": website_section,
    })

    return response.content  # Extract the generated cover letter


# --- FUNCTION TO SEND EMAIL ---
def send_email(sender_email, sender_password, recipient_email, subject, body, resume_path):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Attach cover letter
    msg.attach(MIMEText(body, 'plain')) 

    # Attach resume
    if resume_path:
        with open(resume_path, 'rb') as resume_file:
            resume = MIMEBase('application', 'octet-stream')
            resume.set_payload(resume_file.read())
        encoders.encode_base64(resume)
        resume.add_header(
            'Content-Disposition',
            f'attachment; filename={os.path.basename(resume_path)}'
        )
        msg.attach(resume)

    # Send email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"✅ Email sent to {recipient_email} for position {subject}")
    except Exception as e:
        print(f"❌ Failed to send email to {recipient_email}. Error: {e}")

# --- MAIN SCRIPT ---
try:
    # Load Excel File
    emails_df = pd.read_excel(XL_FILE)
    print(f"Loaded {len(emails_df)} email addresses from {XL_FILE}.")
except FileNotFoundError:
    print(f"Excel file '{XL_FILE}' not found.")
    exit()

# Check if resume exists
if not os.path.isfile(RESUME_PATH):
    print(f"Resume file '{RESUME_PATH}' not found.")
    exit()

# Process each email
for index, row in emails_df.iterrows():
    recipient_email = row['Email']
    company_name = row['Company Name']
    job_position = row['Job Position']
    job_description = row['Job Description']
    company_website = row['Website']

    # Fetch website information (Optional)
    website_info = fetch_website_info(company_website) if company_website else "No website available."

    # Generate AI cover letter
    print(f"📝 Generating cover letter for {job_position} at {company_name}...")
    cover_letter = generate_cover_letter(company_name, job_position, job_description, website_info, resume_text)

    # Construct subject
    subject = f"Job Application for {job_position}"

    # Send email
    send_email(SENDER_EMAIL, SENDER_PASSWORD, recipient_email, subject, cover_letter, RESUME_PATH)

print("\n🚀 All emails have been sent successfully!")
