import logo from './logo.svg';
import './App.css';
import react, { useEffect, useState } from 'react';
function App() {
  const [Name,SetName]=useState("Null");
  const [Age,SetAge]=useState("Number");
  //https://flaskapp-backend.vercel.app
  useEffect(()=>{
   fetch('http://127.0.0.1:5000/',{
      method:'GET'
    }
  ).then((res) => {
     console.log(res.json()
     .then((data)=>
     {
      console.log(data);
      SetName(data.Name);
      SetAge(data.Age)
     }
  ))})
  },[]);
  return (
    <div className="App">
        <p>{Name}</p>
        <p>{Age}</p>
        <p>Hello {Name}</p>
    </div>
  );
}

export default App;
