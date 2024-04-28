var display = 0;

function hideshow(){
    if (display == 1){
        dispatchEvent.style.display = 'block';
        display= 0;
    }

    else{
        dispatchEvent.style.display = 'none';
        display = 1;
    }
}