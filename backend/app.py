from flask import Flask,request,redirect,session,jsonify
from flask_cors import CORS
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
import bcrypt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import spacy
import en_core_web_sm
from spacy.matcher import Matcher
import docx2txt
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
import locationtagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import urllib.request
from werkzeug.utils import secure_filename
from fpdf import FPDF
from PyPDF2 import PdfFileMerger
from nltk.tokenize import RegexpTokenizer
#Frontend and Backend Connector
app=Flask(__name__)
CORS(app)

def create_pdf(input_file):
    # Create a new FPDF object
    pdf = FPDF()

    # Open the text file and read its contents
    with open(input_file, 'r') as f:
        text = f.read()

    # Add a new page to the PDF
    pdf.add_page()

    # Set the font and font size
    pdf.set_font('Arial', size=12)

    # Write the text to the PDF
    pdf.write(5, text)

    # Save the PDF
    return pdf.output('output.pdf')

#Data Cleaning of the  resume
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text

def cos_compute(text1,text2):
        content =[text2,text1]
        countVector= CountVectorizer()
        matrix = countVector.fit_transform(content)
        similarity_matrix=cosine_similarity(matrix)
        print('Strength of the resume based on Job Description:'+str(round((similarity_matrix[1][0]),1)*100)+'%')
        return str(int(round((similarity_matrix[1][0]),1)*100+20))
    
def skill_compute(text1,text2):
    #text1 is resume
        #Computer Science and Engineering Skills
    tokenizer = RegexpTokenizer(r'\w+')
    print(tokenizer.tokenize(text1))
    skill1=pd.read_csv("raw_skills.csv")
    skill10=pd.read_csv("Skills.csv")
    skill101=pd.read_csv("related_skills.csv")
    
    
    df=str(pd.DataFrame(skill1["raw_skills"])).lower()
    x=set(re.findall("skills",df))
    print("skill1 Matching")
    print(x)

    df=str(pd.DataFrame(skill10["Example"])).lower()
    print(df)
    x=set(re.findall(text1,df))
    print("skill10 Matching")
    print(x)

    y=[]
    related0=str(pd.DataFrame(skill101["related_0"])).lower()
    related1=str(pd.DataFrame(skill101["related_1"])).lower()
    related2=str(pd.DataFrame(skill101["related_2"])).lower()
    related3=str(pd.DataFrame(skill101["related_3"])).lower()
    related4=str(pd.DataFrame(skill101["related_4"])).lower()
    related5=str(pd.DataFrame(skill101["related_5"])).lower()
    related6=str(pd.DataFrame(skill101["related_6"])).lower()
    related7=str(pd.DataFrame(skill101["related_7"])).lower()
    related8=str(pd.DataFrame(skill101["related_8"])).lower()
    related9=str(pd.DataFrame(skill101["related_9"])).lower()
    related10=str(pd.DataFrame(skill101["related_10"])).lower()
    y.append(related0)
    y.append(related1)
    y.append(related2)
    y.append(related3)
    y.append(related4)
    y.append(related5)
    y.append(related6)
    y.append(related7)
    y.append(related8)
    y.append(related9)
    y.append(related10)


    #Mechanical Engineering
    skill2=pd.read_csv("Mechanicalskills.csv")
    #Biomedical Engineering
    skill3=pd.read_csv("Biomedical-Engineering.csv")
    #Civil Engineering
    skill4=pd.read_csv("Civil-Engineering.csv")
    #Electrical Communication Engineering
    skill5=pd.read_csv("Electrical-Communication-Engineering.csv")
    #Electrical Engineering
    skill6=pd.read_csv("Electrical-Engieering-Skills.csv")

    sk2=str(pd.DataFrame(skill2["Skills"])).lower()
    sk3=str(pd.DataFrame(skill3["Skills"])).lower()
    sk4=str(pd.DataFrame(skill4["Skills"])).lower()
    sk5=str(pd.DataFrame(skill5["Skills"])).lower()
    sk6=str(pd.DataFrame(skill6["Skills"])).lower()

    z=[sk2,sk3,sk4,sk5,sk6]
    print("skill101 Matching")
    for i in range(0,len(z)):
        x=set(re.findall(text1,z[i]))
        if x:
            print(x)
            break
    return "Skill"
##MONGODB DATABASE CONNECTION
client=pymongo.MongoClient("mongodb+srv://nirmalnj2003:Nirmal123@cluster0.pkba7ij.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    db=client.get_database("ResumeParser")
    records=db.Users
    df = pd.read_csv('UpdatedResumeDataSet.csv')
    print(df.head())

    #Plot for the  no of skills in the resume 
    plt.figure(figsize=(15,5))
    sns.countplot(df['Category'])
    plt.xticks(rotation=90)
    plt.show()

    #words into categorical values
    le = LabelEncoder()
    le.fit(df['Category'])
    df['Category'] = le.transform(df['Category'])

    df.Category.unique()


    #Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(df['Resume'])
    requredText  = tfidf.transform(df['Resume'])

    #Splitting
    X_train, X_test, y_train, y_test = train_test_split(requredText, df['Category'], test_size=0.2, random_state=42)
    X_train.shape
    X_test.shape

    #train the model and print the classification report
    #RandomForestClassifier
    clf = RandomForestClassifier(n_estimators = 20,random_state=42)  
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy score:",accuracy_score(y_test,y_pred))
    classification_rep = classification_report(y_test, y_pred)
    print("\n",classification_rep)

    #Confusion Matrix
    actual = y_test
    predicted = y_pred
    cm = confusion_matrix(actual,predicted)
    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=['0', '1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'],
                yticklabels=['0', '1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'])
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()
    #Prediction System
    pickle.dump(tfidf,open('tfidf.pkl','wb'))
    pickle.dump(clf, open('clf.pkl', 'wb'))
            
except Exception as e:
    print(e)
    
def extraction(resume_text,resume_app):
    
    
    #Name Extraction
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    nlp_text = nlp(resume_text)
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('NAME',[pattern])
    matches = matcher(nlp_text)
    name=''
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        name=span.text
        break
    print('Name:',name)
    resume_app.append(name)
    
    #Phone Number Extraction
    Mobilenumber=''
    phone = re.findall('\d{2}\d{3}\d{5}',resume_text)
    if phone:
        number = ''.join(phone[0])
        if len(number) > 10:
            Mobilenumber='NULL'
        else:
            Mobilenumber=number
    print('Mobile Number: ',Mobilenumber)
    resume_app.append(Mobilenumber)
    #Emailid Extraction
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    r= r.findall(resume_text)
    print('Mail id: ',r)
    resume_app.append(r)
    #Education Qualification
        # Grad all general stop words
    STOPWORDS = set(stopwords.words('english'))

        # Education Degrees
    EDUCATION = [
            'BE','B.E.', 'B.SC', 'BS', 'B.S', 
            'ME', 'M.E', 'M.E.', 'M.B.A', 'MBA', 'MS', 'M.S', 
            'BTECH', 'B.TECH', 'M.TECH', 'MTECH', 
            'SSLC', 'SSC' 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
        ]
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]
    

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index]
                
    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{})))'), edu[key])
        if year:
           education.append((key, ''.join(year[0])))
        else:
           education.append(key)
    print('Qualification: ',education)
    resume_app.append(education)
    
    #Location Extarction
    place_entity = locationtagger.find_locations(text=resume_text)
    #the country
    try:
        if place_entity.countries[0]!=IndexError:
            print("The country is :",place_entity.countries[0])
            resume_app.append(place_entity.cuntries[0])
    except:
         resume_app.append("")
         print("NO COUNTRY")
    #the state
    try:
        if place_entity.regions[0]!=IndexError:
            print("The state is: ",place_entity.regions[0])
            resume_app.append(place_entity.regions[0])
    except:
         resume_app.append("")
         print("NO REGIONS")
    try:
        if place_entity.cities[0]!=IndexError:
        #the city
            print("The city is: ",place_entity.cities[0])
            resume_app.append(place_entity.cities[0])
    except:
        resume_app.append("")
        print("NO CITY")
    
    #Resume Prediction Category
    # Load the trained classifier
    clf = pickle.load(open('clf.pkl', 'rb'))

    # Clean the input resume
    cleaned_resume = cleanResume(resume_text)
    print(cleaned_resume)
    # Transform the cleaned resume using the trained TfidfVectorizer
    input_features = tfidf.transform([cleaned_resume])

    # Make the prediction using the loaded classifier
    prediction_id = clf.predict(input_features)[0]

    # Map category ID to category name
    category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "Software Developer",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and fitness",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
    }

    category_name = category_mapping.get(prediction_id, "Unknown")

    print("Predicted Category:", category_name)
    print(prediction_id)
    resume_app.append(prediction_id)
    resume_app.append(category_name)
    #Resume Strength based on Job Description
    resume_app.append(cos_compute(resume_text,"AI and ML.txt"))












@app.route('/',methods=['POST','GET'])
def HomePage():
    return{
        "Name":"Nirmal",
        "Age":21
    }

@app.route('/register',methods=['POST'])
def RegisterPage():
     if session.get("EmailId"):
        return "You have already LoggedIn"
     if request.method == 'POST':
        data=request.json
        print(type(data))
        user=data["Name"]
        email=data["EmailId"]
        password=data["Password"]
        confirmpassword=data["Confirmpassword"]
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        Registered_details={"Name":user,"EmailId":email,"Password":password,"Password":hashed}
        records.insert_one(Registered_details)
        return "Registered "+user
     return {
            "Status":"Success",
            "Name":user,
            "EmailId":email,
            "Password":password,
            "Confirmpassword":confirmpassword
            }
@app.route('/login',methods=['POST'])
def LoginPage():
    if request.method=='POST':
        data=request.json
        print(type(data))
        email=data["EmailId"]
        password=data["Password"]
        print(email)
        user=records.find_one({"EmailId":email})
        try:
            u1_email=user['EmailId']
            u1_password=user['Password']
            u1_id=str(user['_id'])
            if u1_email:
                if bcrypt.checkpw(password.encode('utf-8'), u1_password):
                   return{
                       "Result":"EmailId and Password Matched",
                       "id":u1_id
                   }
                else:
                   return{
                    "Result":"Password not Matched"
                }
            else:
                return{
                    "Result":"EmailId not Matched"
                }
        except:
            return{
                "Result":"Invalid EmailId "
            }
    return{
         "Result":"EmailId and Password Matched"
    }
@app.route('/getuserinfo',methods=['POST'])
def getuserinfo():
    if request.method=='POST':
       data=request.json
       id=data['id']
       objins=ObjectId(id)
       print(objins)
       user=records.find_one({"_id":objins})
       u1_id=str(user['_id'])
       u1_name=user['Name']
       return{
            "Result":u1_name
        }
    return{
        "Result":"No Id"
    }
    





app.secret_key = "caircocoders-ednalan"
  
UPLOAD_FOLDER = 'static/uploads'
JOB_FOLDER='static/jobs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JOB_FOLDER'] = JOB_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['txt', 'pdf','doc'])



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploadedfile',methods=['POST'])
def uploadedfile():
    resume_app=[]
    if request.method == 'POST':
       
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resume_text=extract_text_from_pdf('static/uploads/'+filename)
            extraction(resume_text,resume_app)
            print(resume_app)
            return {
                "Name":str(resume_app[0]),
                "Mobile Number":str(resume_app[1]),
                "MailId":str(resume_app[2]),
                "Qualification":str(resume_app[3]),
                "Country":str(resume_app[4]),
                "State":str(resume_app[5]),
                "City":str(resume_app[6]),
                "PredictedId":str(resume_app[7]),
                "PredictedName":str(resume_app[8]),
                "ResumeStrength":str(resume_app[9])
            }

@app.route('/jobfile',methods=['POST'])
def jobfile():
    if request.method == 'POST':
        if 'file' not in request.files and 'file1' not in request.files:
            print('No file part')
            return redirect(request.url)
        file1 = request.files['file']
        file2 = request.files['file1']
        print(file1)
        print(file2)
        if file1.filename == '' and file2.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            file1.save(os.path.join(app.config['JOB_FOLDER'], filename1))
            file2.save(os.path.join(app.config['JOB_FOLDER'], filename2))
            job_text1=extract_text_from_pdf('static/jobs/'+filename1)
            create_pdf('static/jobs/'+filename2)
            job_text2=extract_text_from_pdf('output.pdf')
            str_resume_job=cos_compute(job_text1,job_text2)
            str_resume_skill=skill_compute(job_text1,job_text2)
            return{ 
               "Status":"Success",
               "Job1":job_text1,
               "Job2":job_text2,
               "str_resume_job":str_resume_job,
               "str_resume_skill":str_resume_skill
            }
        return {
          "Status":"Failure" 
        }
    
if __name__=="__main__":
    app.run(debug=True)
 