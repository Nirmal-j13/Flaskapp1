import React, { useState } from 'react'
import '../Uploadfiledemo/Uploadfiledemo.css'
import { ToastContainer, toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';
export const Uploadfiledemo = ({uploadedornot}) => {
    const[selectedfile,setselectedfile]=useState(null);
    const[resume,setresume]=useState(false);
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
         alert(data.errors)
      }
      else {
         console.log(data)
		 uploadedornot(true);
      }
   })
   {
	resume?
    <div>  
	{toast.done("See Your Outcome", {
	position:"top-right"
	})}
</div>:
<div>  
	{toast.loading("UploadingFile", {
	position:"top-right"
	})}
</div>
}
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
    return (	
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
        
    );
}
