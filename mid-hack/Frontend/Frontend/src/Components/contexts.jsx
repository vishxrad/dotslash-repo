import React from "react";

export default function ContextPanel({ contexts }) {
  return (
    <div>
      <div className="bg-white/50 overflow-y-scroll no-scrollbar w-96 h-[460px] rounded-md">
        <div className="w-full max-w-md mx-auto p-4">
          <h1 className="text-white text-4xl pb-4 text-center font-semibold">
            Contexts
          </h1>
          <div className="border rounded-lg py-6 shadow-sm">
            <div className="h-96 no-scrollbar overflow-y-auto p-4">
              {contexts.map((box) => (
                <div
                  key={box.id}
                  className={`${box.color} p-4 rounded-lg mb-4 last:mb-0 text-slate-400 transition-transform hover:scale-[1.05]`}
                >
                  <h3 className="font-semibold mb-2">{box.title}</h3>
                  <p>{box.content}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
