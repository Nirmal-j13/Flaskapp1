import axios from "axios";
import React, { Component } from "react";
import './Uploadfile.css'
import { ToastContainer, toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';
class Uploadfile extends Component {
	
	state = {
		selectedFile: null,
	};
	resumestate={
		resumeDatas:false
	}
	
	onFileChange = (event) => {
		this.setState({
			selectedFile: event.target.files[0],
		});
	};
	onFileUpload =(e)=>  {
		e.preventDefault()
   let file = this.state.selectedFile
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
		 this.setState({resumestate:true})
		console.log(this.state.resumeDatas)
      }
   })
   {
	this.resumestate.resumeDatas?<div>  
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
	fileData = () => {
		if (this.state.selectedFile) {
			return (
				<div>
					<b><h2>File Details</h2></b>
					<p>
						<b>File Name:{" "}</b>
						{this.state.selectedFile.name}
					</p>

					<p>
                        <b>File Type:{" "}</b>
						{this.state.selectedFile.type}
					</p>

					<p>
						<b>Last Modified:{" "}</b>
						{this.state.selectedFile.lastModifiedDate.toDateString()}
					</p>
				</div>
			);
		} 
	};

	render() {
		return (
				
			<div>
					<input
						type="file"
						onChange={this.onFileChange}
					/>
					<button onClick={this.onFileUpload} className="topcontent__uploadbutton ">
						Upload File
					</button>
				{this.fileData()}
			</div>
			
		);
	}
}

export default Uploadfile;
