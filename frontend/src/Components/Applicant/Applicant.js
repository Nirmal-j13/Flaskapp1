import React, { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import './Applicant.css'
import { useNavigate } from 'react-router-dom';
import { ToastContainer, toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';
import resumeimg from '../Asset/resume.jpg';
import { LoadingBar } from '../LoadingBar/LoadingBar';
import { CircularProgressbar } from 'react-circular-progressbar';
import { buildStyles } from 'react-circular-progressbar'
import 'react-circular-progressbar/dist/styles.css';

export const Applicant = () => {
const [name,setname]=useState("");
const [load,setload]=useState(false);
const[selectedfile,setselectedfile]=useState(null);
const[jobfile,setjobfile]=useState(null);
const[resumeimage,setresumeimage]=useState("imagenotloaded");
const[resume,setresume]=useState(false);
const[job,setjob]=useState(false);
const[fileupload,setfileupload]=useState(false);
const[loadimg,setloadimg]=useState(false);
const[resumetoast,setresumetoast]=useState(false)
const navigate = useNavigate();
let { id } = useParams();
let formData = new FormData()
useEffect(()=>{
     fetch('http://127.0.0.1:5000/getuserinfo',{
      method:"POST",
      body:JSON.stringify({
        id:id
      }),
      headers: {
        "Content-type": "application/json; charset=UTF-8"
    }
     }).then((data)=>{console.log(data.json().then((data)=>{console.log(data);setname(data.Result)}))})
},[]);
const onJobFileChange =(event)=>{
  setjobfile(event.target.files[0])
}
const onFileChange = (event) => {
    setselectedfile(event.target.files[0])
};
const onFileUpload =(e)=>  {
e.preventDefault()
let file = selectedfile
formData.append('file', file)
console.log(formData);
fetch('http://127.0.0.1:5000/uploadedfile', {
  method: 'POST',
  body: formData,
},
)
.then(resp => resp.json())
.then(data => {
  if (data.errors) {
    <div>  
      {toast.error(data, {
        position:"top-right"
      })}
    </div>
     alert(data.errors)
  }
  else {
     console.log(data)
     setresume(data)
    
      return(
        <div>  
        {toast.success("Uploaded Successfully", {
          position:"top-right"
        })}
      </div>)
  }
})
};
const fileData = () => {
if (selectedfile) {
    return (
        <div>
            <b><h2>File Details</h2></b>
            <p>
                <b>File Name:{" "}</b>
                {selectedfile.name}
            </p>

            <p>
                <b>File Type:{" "}</b>
                {selectedfile.type}
            </p>

            <p>
                <b>Last Modified:{" "}</b>
                {selectedfile.lastModifiedDate.toDateString()}
            </p>
        </div>
    );
} 
};
const onJobFileUpload =(e)=>  {
  e.preventDefault()
  let file1 = jobfile
  formData.append('file',selectedfile)
  formData.append('file1', file1)
  console.log(formData.has('file'));
  console.log(formData.has('file1'));

  console.log("Uploaded Resume file:"+selectedfile)
  console.log("Uploaded job file:"+file1)
  fetch('http://127.0.0.1:5000/jobfile', {
    method: 'POST',
    body:formData
  })
  .then(resp => resp.json())
  .then(data => {
    if (data.errors) {
      <div>  
        {toast.error(data, {
          position:"top-right"
        })}
      </div>
       alert(data.errors)
    }
    else {
       console.log(data)
       setjob(data)
        return(
          <div>  
          {toast.success("Job File Uploaded", {
            position:"top-right"
          })}
        </div>)
    }
  })
  };
    return(
      <>
        {name===""&&load==false?<LoadingBar/>
        :
        <div>
        <div className='Navbar-bg'>
        <ToastContainer/>
          <div className='Navbar-left'>
              <span className='Navbar-logo'>P</span>arse<span className='Navbar-logo'>P</span>ros<span className='Navbar-logo'>X</span>
          </div>
          <div className='Navbar-right'>
       <Link to='/' smooth={true} duration={500} className='Navbar-link'>
          <h4>
              Home
          </h4>
         </Link>
         <Link to={'/sigin/'+id+'/applyjob'} smooth={true} duration={500} className='Navbar-job'>
            <h4>
                <b>A</b>pply<b>J</b>ob
            </h4>
           </Link>
         <h4 className='Navbar-link' onClick={()=>{   setInterval(()=>{
                navigate('/')
                clearInterval()
            },3000)
            return(
            <div>  
            {toast.success("LoggedOut Successful", {
              position:"top-right"
            })}
          </div>)}}>
             Signout
         </h4>

         </div>   
       </div>
       <div className='applicant'>
        <div className='applicant-left'>
        {resume?
         job?
          <div>
               <p className='ResumeStrength'>
                  The Resume Strength based on Job Description
              </p>
              <CircularProgressbar 
                  className='applicant-circularbar'
                  strokeWidth={10}
                  value={job.str_resume_job} 
                  text={`${job.str_resume_job}%`}
                  styles={buildStyles({
                    textSize: '10px',
                    pathTransitionDuration: 0.5,
                    textColor: '#000000',
                    backgroundColor: '#173342',
                    boxshadow:'0px 5px 16px 4px rgb(141, 255, 249)'
                  })} 
                 />
          <div className='applicant-resume'>
             <p className='resume'>
               <h1 className='resume-h1'>Predicted Resume Strength:</h1>
               <span className='resume-pred'>{job.str_resume_job}%</span>
             </p>
             <p className='resume'>
               <h1 className='resume-h1'>Resume Key Skills Matched:</h1>
               <span className='resume-pred'>{job.str_resume_job}%</span>
             </p>
          </div>
          </div>
          :
         <div>
          <p className='ResumeStrength'>
              The Resume Strength and Predicted Category
          </p>
          <CircularProgressbar 
                  className='applicant-circularbar'
                  strokeWidth={10}
                  value={resume.ResumeStrength} 
                  text={`${resume.ResumeStrength}%`}
                  styles={buildStyles({
                    textSize: '10px',
                    pathTransitionDuration: 0.5,
                    textColor: '#000000',
                    backgroundColor: '#173342',
                    boxshadow:'0px 5px 16px 4px rgb(141, 255, 249)'
                  })} 
                 />
          <div className='applicant-resume'>
             <p className='resume'>
               <h1 className='resume-h1'>Predicted Id:</h1>
               <span className='resume-pred'>{resume.PredictedId}</span>
             </p>
             <p className='resume'>
               <h1 className='resume-h1'>Predicted Resume Strength:</h1>
               <span className='resume-pred'>{resume.ResumeStrength}%</span>
             </p>
             <p className='resume'>
               <h1 className='resume-h1'>Predicted Category:</h1>
               <span className='resume-pred'> {resume.PredictedName}</span>
             </p>
             <p className='applicant-text-btn'>
              Parse the Resume Based on Job Description (Optional)
              </p>
              <input
                    type="file"
                    onChange={onJobFileChange}
                />
              <button className='applicant-job-btn' onClick={onJobFileUpload}>Upload Job Description</button>
          </div> 
          </div>
           :
            <div className='applicant-left-initial'>                    
              <img src={resumeimg} alt={resumeimg} height={500} width={500}  />
            <div>
              <p className='applicant-text'>Once You upload the resume,Your Resume Prediction will be revealed here</p>
            </div>
          </div>
        }
      </div>
         <div className='topcontent'>
          <div className='topcontent__container'>       
              <h1 className='hello'><b>Hello {name}!</b></h1>       
            <h2>Welcome to ParseprosX.com</h2>
            <p className='applicant-text'>Upload your Resume</p>
            <div>
                <input
                    type="file"
                    onChange={onFileChange}
                />
                <button onClick={onFileUpload} className="topcontent__uploadbutton ">
                    Upload File
                </button>
                {fileData()}
        </div>
          </div>
        </div>
       </div>
       </div>
        }
      </>
    )
}
