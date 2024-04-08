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
const[resumeimage,setresumeimage]=useState("imagenotloaded");
const[resume,setresume]=useState(false);
const[fileupload,setfileupload]=useState(false);
const[loadimg,setloadimg]=useState(false);
const navigate = useNavigate();
let { id } = useParams();
console.log(typeof(id))
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

const onFileChange = (event) => {
    setselectedfile(event.target.files[0])
};
const onFileUpload =(e)=>  {
e.preventDefault()
let file = selectedfile
let formData = new FormData()
formData.append('file', file)
console.log(formData)
fetch('http://127.0.0.1:5000/uploadedfile', {
  method: 'POST',
  body: formData
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
          </div>
          </div>:
            <div className='applicant-left-initial'>                    
              <img src={resumeimg} alt={resumeimg} height={500} width={500}  />
            <div>
              <p className='applicant-text'>Your Resume Prediction will be revealed here, Once You upload the resume</p>
            </div>
          </div>
        }
      </div>
         <div className='topcontent'>
          <div className='topcontent__container'>       
              <h1 className='hello'><b>Hello {name}!</b></h1>       
            <h2>Welcome to ParseprosX.com</h2>
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
