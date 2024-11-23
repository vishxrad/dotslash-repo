// App.jsx
import React, { useState, useEffect } from "react";
import SearchArea from "./Components/SearchArea";
import ContextPanel from "./Components/contexts";

function App() {
  const [contexts, setContexts] = useState([]);
  const [text, setText] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState(null);

  useEffect(() => {
    if (!("webkitSpeechRecognition" in window)) {
      console.error("Speech recognition not supported in this browser");
      return;
    }

    const recognitionInstance = new window.webkitSpeechRecognition();
    recognitionInstance.continuous = true;
    recognitionInstance.interimResults = true;

    recognitionInstance.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join("");
      setText(transcript);
    };

    recognitionInstance.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      setIsListening(false);
    };

    setRecognition(recognitionInstance);

    return () => {
      if (recognitionInstance) {
        recognitionInstance.stop();
      }
    };
  }, []);

  const toggleListening = () => {
    if (!recognition) return;

    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
    }
    setIsListening(!isListening);
  };

  const MicButton = () => (
    <button className="font-mono active:bg-[rgba(0,0,0,0)] bg-white mt-2 pb-2 w-10 text-white rounded-md">
      <div
        onClick={toggleListening}
        className="bg-blue-300 border-[1.5px] border-white transition duration-100 pt-2 active:translate-y-2 pb-2 rounded-md hover:shadow-lg hover:shadow-gray-50 active:shadow-gray-50/0 pl-[7px]"
      >
        <img
          src={
            isListening
              ? "https://www.svgrepo.com/show/449835/microphone-slash.svg"
              : "https://www.svgrepo.com/show/449834/microphone.svg"
          }
          alt={isListening ? "Stop recording" : "Start recording"}
          className="p-1 w-6 h-6"
        />
      </div>
    </button>
  );

  return (
    <div className="mx-auto my-16 flex flex-row gap-6 flex-wrap justify-center items-center">
      <ContextPanel contexts={contexts} />
      <div className="bg-white/50 p-10 rounded-md">
        <SearchArea
          voiceInput={text}
          contexts={contexts}
          setContexts={setContexts}
          micButton={<MicButton />}
        />
      </div>
    </div>
  );
}

export default App;
