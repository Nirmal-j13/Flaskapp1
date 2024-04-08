import React, { useState } from 'react'
import './Register.css'
import logo from '../Asset/logo.png'
import { ToastContainer, toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';
import PasswordStrengthBar from 'react-password-strength-bar';
import { useNavigate } from 'react-router-dom';
import validator from 'validator'
export const Register = () => {
  const [name,setname]=useState("");
  const [email,setemail]=useState("");
  const [password,setpassword]=useState("");
  const [confirmpassword,setconfirmpassword]=useState("");
  const [match,setmatch]=useState(false);
  const navigate = useNavigate();
  const OnHandleSubmit=async(e)=>{  
     try{
      e.preventDefault();
        if(name.length<=2)
        { 
          console.log(name);
          return(
             <div>  
              {toast.warning('Invalid Name/No Name', {
                position:"top-right"
              })}
            </div>
          )
        }
        if(!validator.isEmail(email))
        {
          return(
            <div>  
             {toast.warning('Invalid EmailId/No EmailId', {
               position:"top-right"
             })}
           </div>
         )
        }
        else
        {
        await fetch("http://127.0.0.1:5000/register",{
          method:"POST",
          body:JSON.stringify({
            Name:name,
            EmailId:email,
            Password:password,
            Confirmpassword:confirmpassword
          }),
          headers: {
            "Content-type": "application/json; charset=UTF-8"
        }
        }).then(()=>{

          return(
            <div>  
             {toast.success('Registered Successfully', {
               position:"top-right"
             })}
           </div>
         )
        }).then(()=>{
          setInterval(()=>{
            navigate('/login');
           },3000)
        })
      }
    }
    catch(err)
    {
      console.log(err);
    }
  }
  return (
    <div className='signup-bg'>
         <div className="signup-splitleft">
              <div className='signup-1'>
                <div className='signup-img'>
                <img src={logo} style={{height:"100px", width:"100px"}} alt='logo'/>
                </div>
                <div className='signup-2'>
                   <span className='Navbar-logo'>P</span>arse<span className='Navbar-logo'>P</span>ros<span className='Navbar-logo'>X</span>
                </div>
                <small style={{display:"flex", alignItems:"center",justifyContent:"center",color:"black"}}>The Smart Automated Intelligent System</small>
              </div>
         </div>
         <div className="signup-splitright">
            <div className='signup-signup'>
                <h2><i>Welcome to the ParseprosX.com</i></h2>
                <em><b>Create an Account</b></em>
                <form className='signup-form'>
                <ToastContainer/>
                   <label>Name</label>
                 <input type='text'
                        placeholder='Name'
                        value={name}
                        onChange={(e)=>{setname(e.target.value)}}
                        />
                  <label>Email</label>
                 <input type='email'
                        placeholder='Email'
                        value={email}
                        onChange={(e)=>{setemail(e.target.value)}}
                        />
                  <label>Password</label>
                 <input type='password'
                        placeholder='Password'
                        value={password}
                        onChange={(e)=>{setpassword(e.target.value)}}
                        />
                  <PasswordStrengthBar password={password} />
                  <label>Confirm Password</label>
                 <input type='password'
                        placeholder='Confirm Password'
                        value={confirmpassword}
                        onChange={(e)=>{
                          confirmpassword===password?setmatch(true):setmatch(false);
                          setconfirmpassword(e.target.value)
                        }}
                        />
                        {match?<div style={{color:"darkgreen", fontWeight:450}}>Matched</div>:<div></div>}
                        <div className='sigin-aa'>
                    <small>
                      <p onClick={()=>{
                         setInterval(()=>{
                            navigate('/signin')
                         },1000)
                      }}>Already have an Account</p>
                    </small>
                 </div>
                 <button className='signup-link' onClick={OnHandleSubmit} >
                  Signup
                  </button>      
                  </form>
            </div>
         </div>

    </div>
  )
}
