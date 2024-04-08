import React from 'react'
import './Navbar.css'
import {Chart as ChartJS,BarElement, CategoryScale, LinearScale} from 'chart.js'
import {Bar} from 'react-chartjs-2'
import { Link } from 'react-router-dom'
import resumeprof from '../Asset/resumeprofile.png';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement
)
export const Navbar = () => {
  
  const data = {
    labels:["Java Developer",
    "Testing",
    "DevOps Engineer",
    "Python Developer",
    "Web Designing",
    "HR",
    "Hadoop",
    "Blockchain",
    "ETL Developer",
    "Operations Manager",
    "Data Science",
    "Sales",
    "Mechanical Engineer",
    "Arts",
    "Database",
    "Electrical Engineering",
    "Health and fitness",
    "PMO",
    "Business Analyst",
    "DotNet Developer",
    "Automation Testing",
     "Network Security Engineer",
    "SAP Developer",
    "Civil Engineer",
    "Advocate",],
    datasets: [{
      label: 'My First Dataset',
      data: [84, 70, 55, 48, 45, 44, 42,40,40,40,40,40,40,36,33,30,30,30,28,28,26,25,24,24,20],
      backgroundColor: [
        'rgba(255, 99, 132, 0.2)',
        'rgba(255, 159, 64, 0.2)',
        'rgba(255, 205, 86, 0.2)',
        'rgba(75, 192, 192, 0.2)',
        'rgba(54, 162, 235, 0.2)',
        'rgba(153, 102, 255, 0.2)',
        'rgba(201, 203, 207, 0.2)'
      ],
      borderColor: [
        'rgb(255, 99, 132)',
        'rgb(255, 159, 64)',
        'rgb(255, 205, 86)',
        'rgb(75, 192, 192)',
        'rgb(54, 162, 235)',
        'rgb(153, 102, 255)',
        'rgb(201, 203, 207)'
      ],
      borderWidth: 1.5
    }]
  };
  var options = {
    maintainAspectRatio: false,
    scales: {
    },
    legend: {
      labels: {
        fontSize: 25,
      },
    },
  }
  return (
    <>
          <div className='Navbar-bg'>
            <div className='Navbar-left'>
                <span className='Navbar-logo'>P</span>arse<span className='Navbar-logo'>P</span>ros<span className='Navbar-logo'>X</span>
            </div>
            <div className='Navbar-right'>
         <Link to='/' smooth={true} duration={500} className='Navbar-link'>
            <h4>
                Home
            </h4>
           </Link>
           <Link to='/signin' smooth={true} duration={500} className='Navbar-link'>
            <h4>
                Signin
            </h4>
           </Link>
           <Link to='/signup' smooth={true} duration={500} className='Navbar-link'>
            <h4>
                Signup
            </h4>
           </Link>
           </div>   
         </div>
         <div className='home-bg'>
           <div className='home-left'>
              <p className='home-heading'>The Divine Power of </p>
              <p className='home-heading'>High Quality Resume</p>

              <p className='home-heading-1'>The Accuracy and Parsing denotes Strength Quality of the Resume</p>
              <p className='home-heading-2'>
                ParseProsX.com is one of the best smart Intelligent platform for analysing the individual resume strength 
                and predicting the resume category based on various aspect of skills. It have Provided the accuracy rate of 
                99.67% in training datas and 67.64% in testing datas which have been developed by using the <b>Random Forest
                Classifier Algorithm.</b>  
              </p>
           </div>
           <div className='home-right'>
                <img src={resumeprof} alt='profile' height={500} width={500}/>
           </div>
         </div>
         <div className='Resume-Skill-Category'>
             <p className='p-Resume-Skill-Category'>Resume Skill Category</p>
             <p className='home-heading-1'>X-Axis Shows the type of the Resume Which are Trained</p>
             <p className='home-heading-1'>Y-Axis Shows the Total no of individual Resume</p>
          </div>
         <div className='Bar-Chart'>
               <Bar
                 data={data}
                 height={400}
                 width={400}
                 options={options}
               />
              </div>
    </>
  )
}
