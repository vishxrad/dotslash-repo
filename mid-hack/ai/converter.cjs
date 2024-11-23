const fs = require("fs");
const csv = require("csv-parser");
const createCsvWriter = require("csv-writer").createObjectCsvWriter;

// Function to transform CSV data
async function transformCsv(inputFile, outputFile) {
  try {
    const results = [];

    // Define the CSV writer with new headers
    const csvWriter = createCsvWriter({
      path: outputFile,
      header: [
        // Define your output headers here
        { id: "question", title: "question" },
        { id: "answer", title: "answer" },
        { id: "context", title: "context" },
        { id: "ticker", title: "ticker" },
        { id: "filing", title: "filing" },
      ],
    });

    // Read and process the input CSV file
    await new Promise((resolve, reject) => {
      fs.createReadStream(inputFile)
        .pipe(csv())
        .on("data", (row) => {
          // Transform each row as needed
          const transformedRow = {
            // Example transformation - adjust according to your needs
            question: row.Question, // Combining first and last names
            answer: row.Answer,
            context: `medical question about ${row.qtype}`,
            ticker: "NVDA",
            filing: "2023_10K", // Converting city to uppercase
          };
          results.push(transformedRow);
        })
        .on("end", () => {
          resolve();
        })
        .on("error", (error) => {
          reject(error);
        });
    });

    // Write the transformed data to the output file
    await csvWriter.writeRecords(results);
    console.log("CSV file has been transformed and saved successfully");
  } catch (error) {
    console.error("Error transforming CSV:", error);
  }
}

// Example usage
const inputFile = "train.csv";
const outputFile = "output.csv";

transformCsv(inputFile, outputFile);
