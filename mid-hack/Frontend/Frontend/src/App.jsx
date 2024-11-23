import SearchArea from "./Components/SearchArea";
import { useState, useEffect } from "react";
function App() {
  const [text, setText] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState(null);
  useEffect(() => {
    // Check if browser supports speech recognition
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

    // Cleanup
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
    <>
      <div className=" mx-auto my-16 flex flex-col gap-6 flex-wrap justify-center items-center">
        <SearchArea></SearchArea>

        <div className="p-4">
          <div className="mb-4">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full p-2 border rounded-md h-32"
              placeholder="Your speech will appear here..."
            />
          </div>
          <button
            onClick={toggleListening}
            className={`px-4 py-2 rounded-md ${
              isListening
                ? "bg-red-500 hover:bg-red-600"
                : "bg-blue-500 hover:bg-blue-600"
            } text-white`}
          >
            {isListening ? "Stop Listening" : "Start Listening"}
          </button>
        </div>
      </div>
    </>
  );
}

export default App;
