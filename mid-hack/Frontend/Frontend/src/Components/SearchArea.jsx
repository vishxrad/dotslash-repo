// SearchArea.jsx
import React, { useState } from "react";
import { useFetch } from "../Hooks/useFetch";

export default function SearchArea() {
  const [query, setQuery] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const { fetchData, loading, error } = useFetch();

  const handleSubmit = async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append("query", query);
    if (file) {
      formData.append("file", file);
    }

    const response = await fetchData("/api/your-endpoint", {
      method: "POST",
      body: formData,
    });

    if (response) {
      setResult(response.text || JSON.stringify(response));
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col items-center gap-4">
      <textarea
        value={result}
        readOnly
        className="w-96 h-72 rounded-md border-2 border-white p-6 font-mono text-white bg-gradient-to-br from-black via-gray-900 to-black hover:shadow-lg hover:shadow-white/50 hover:scale-105 transition-all duration-200 resize-none focus:outline-none focus:ring-2 focus:ring-white/50"
        placeholder={loading ? "Loading..." : error || "Your Results ..."}
      />

      <div className="flex items-center gap-4 mt-4">
        <div className="flex rounded-md overflow-hidden border border-gray-300">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask Your Query"
            type="text"
            className="w-36 px-3 py-1.5 focus:outline-none"
          />
          <label className="bg-white px-3 py-1.5 cursor-pointer hover:bg-slate-100 transition-colors border-l border-gray-300 flex items-center">
            <input
              type="file"
              onChange={(e) => setFile(e.target.files[0])}
              className="hidden"
            />
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
            </svg>
          </label>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="font-mono active:bg-[rgba(0,0,0,0)] bg-white mt-2 pb-2 w-32 text-white rounded-md"
        >
          <div className="bg-gradient-to-br from-black via-gray-900 to-black border-[1.5px] border-white transition duration-100 pt-2 active:translate-y-2 pb-2 rounded-md hover:shadow-lg hover:shadow-gray-50 active:shadow-gray-50/0">
            {loading ? "Loading..." : "Ask"}
          </div>
        </button>
      </div>
    </form>
  );
}
