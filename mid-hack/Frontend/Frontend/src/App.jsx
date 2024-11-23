import React, { useState, useEffect } from "react";
import SearchArea from "./Components/SearchArea";

function App() {
  const [text, setText] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState(null);

  useEffect(() => {
    if ("webkitSpeechRecognition" in window) {
      const recognitionInstance = new window.webkitSpeechRecognition();
      recognitionInstance.continuous = true;
      recognitionInstance.interimResults = true;

      recognitionInstance.onresult = (event) => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            transcript += event.results[i][0].transcript;
          }
        }
        setText(transcript);
      };

      recognitionInstance.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
      };

      setRecognition(recognitionInstance);
    } else {
      console.error("Speech recognition not supported in this browser");
    }

    return () => {
      if (recognition) {
        recognition.stop();
      }
    };
  }, []);

  const toggleListening = () => {
    if (isListening) {
      recognition.stop();
      setIsListening(false);
    } else {
      recognition.start();
      setIsListening(true);
    }
  };

  return (
    <div className="mx-auto my-16 flex flex-col gap-6 flex-wrap justify-center items-center">
      <SearchArea
        mic={
          <div>
            <button className="font-mono active:bg-[rgba(0,0,0,0)]  bg-white mt-2 pb-2 w-10  text-white rounded-md">
              <div
                onClick={toggleListening}
                className="bg-gradient-to-br from-black via-gray-900 to-black border-[1.5px] border-white transition duration-100 pt-2 active:translate-y-2 pb-2 rounded-md hover:shadow-lg hover:shadow-gray-50 active:shadow-gray-50/0 pl-[7px] 
          
        
        "
              >
                {isListening ? (
                  <img
                    src="https://www.svgrepo.com/show/449835/microphone-slash.svg "
                    alt="mic"
                    className=" p-1 w-6 h-6"
                  />
                ) : (
                  <img
                    src="https://www.svgrepo.com/show/449834/microphone.svg "
                    alt="mic"
                    className=" p-1 w-6 h-6"
                  />
                )}
              </div>
            </button>
          </div>
        }
        voiceInput={text}
      />
    </div>
  );
}

export default App;
