import './App.css';
import { Login } from './Components/Login/Login';
import { Navbar } from './Components/Navbar/Navbar';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { Register } from './Components/Register/Register';
import { Applicant } from './Components/Applicant/Applicant';
function App() {
  
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path='/' element={<Navbar/>}/>
          <Route path='/signin' element={<Login/>}/>
          <Route path='/signup' element={<Register/>}/>
          <Route path='/signin/:id' element={<Applicant/>}/>
        </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;
