import React from 'react'
import ReactLoading from 'react-loading';
import '../LoadingBar/LoadingBar.css'
export const LoadingBar = () => {
    return(
        <div className='Loader'>
        <ReactLoading type="bars" color='lightblue'
      height={200} width={100} />
      </div>
      )
}
