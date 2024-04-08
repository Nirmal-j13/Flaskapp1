import React, { useState } from 'react'
import './Login.css'
import logo from '../Asset/logo.png'
import { ToastContainer, toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';
import { useNavigate } from 'react-router-dom';
import { deployurl } from '../Url';
import ReactLoading from 'react-loading';




export const Login = () => {
  const [email,setemail]=useState("");
  const [password,setpassword]=useState("");
  const [login,setlogin]=useState(false);
  const navigate = useNavigate();
  const OnHandleSubmit=async(e)=>{  
      e.preventDefault();
      try
      {
      await fetch(`${deployurl}/login`,{
          method:"POST",
          body:JSON.stringify({
            EmailId:email,
            Password:password,
          }),
          headers: {
            "Content-type": "application/json; charset=UTF-8"
        }
        }).then((data)=>{
          console.log(data.json().then((data)=>{
            console.log(data)
            if(data.Result=="EmailId and Password Matched")
            {
              console.log("djghj");
              setInterval(()=>{
                  navigate('/signin/'+data.id)
              },3000)
              return(
              <div>  
              {toast.success("LoggedIn Successful", {
                position:"top-right"
              })}
            </div>)
            }
            else 
            {
              return(
                <div>  
                {toast.error(data.Result, {
                  position:"top-right"
                })}
              </div>)
            }
          }));
        })
      }
      catch(err)
      {
        console.log(err)
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
                <em><b>Login First</b></em>
                <form className='signup-form'>
                  <ToastContainer/>
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
                 <small className='signin-fp'><b>forgot password?</b></small>
                 <div className='sigin-aa'>

                      <p  style={{fontSize:"13px"}}onClick={()=>{
                            navigate('/signup')
                      }}>New User</p>
                 </div>
                 <button className='signup-link' onClick={OnHandleSubmit} >
                  Signin
                  </button>      
                  </form>
            </div>
         </div>

    </div>
  )
}
