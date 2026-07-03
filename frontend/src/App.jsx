import { useState } from "react";
import axios from "axios";
import "./App.css";
import ReactMarkdown from "react-markdown";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  const exampleQuestions = [
    "What is HbA1c?",
    "How does dengue spread?",
    "What are symptoms of flu?",
    "How is asthma managed?",
    "What causes high cholesterol?",
    "What are eczema symptoms?",
  ];

  const getSourceName = (url = "") => {
    if (url.includes("cdc.gov")) return "CDC";
    if (url.includes("who.int")) return "WHO";
    if (url.includes("medlineplus.gov")) return "MedlinePlus";
    if (url.includes("niddk.nih.gov")) return "NIDDK";
    if (url.includes("nih.gov")) return "NIH";
    if (url.includes("diabetes.org")) return "American Diabetes Association";
    if (url.includes("clevelandclinic.org")) return "Cleveland Clinic";
    return "Source";
  };

  const askQuestion = async (customQuestion = question) => {
    if (!customQuestion.trim()) {
      alert("Please enter a medical question");
      return;
    }

    setQuestion(customQuestion);
    setLoading(true);
    setAnswer("");
    setSources([]);

    try {
      const response = await axios.post("http://127.0.0.1:8000/ask", {
        question: customQuestion,
      });

      setAnswer(response.data.answer);
      setSources(response.data.sources || []);
    } catch (error) {
      setAnswer("Something went wrong. Please check if FastAPI is running.");
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setQuestion("");
    setAnswer("");
    setSources([]);
    setLoading(false);
  };

  return (
    <div className="app-container">
      <h1>Medical Knowledge Assistant</h1>

      <p className="subtitle">
        Ask educational questions about diabetes, dengue, flu, asthma,
        cholesterol, blood pressure, migraine, food poisoning, COVID-19, and eczema.
      </p>

      <div className="example-buttons">
        {exampleQuestions.map((item, index) => (
          <button
            key={index}
            className="example-button"
            onClick={() => askQuestion(item)}
            disabled={loading}
          >
            {item}
          </button>
        ))}
      </div>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Example: How does dengue spread?"
        rows="4"
      />

      <div className="action-buttons">
        <button onClick={() => askQuestion()} disabled={loading}>
          {loading ? "Thinking..." : "Ask"}
        </button>

        <button className="clear-button" onClick={clearChat} disabled={loading}>
          Clear
        </button>
      </div>

      {loading && (
        <div className="loading-box">
          <div className="spinner"></div>
          <p>Searching trusted medical sources and generating an answer...</p>
        </div>
      )}

      {answer && (
        <div className="answer-box">
          <ReactMarkdown>{answer}</ReactMarkdown>
        </div>
      )}

      {sources.length > 0 && (
        <div className="sources-box">
          <h2>Sources</h2>

          {sources.map((source, index) => (
            <div className="source-card" key={index}>
              <strong>
                Source {index + 1}: {getSourceName(source.source_url)}
              </strong>

              <p>
                <strong>Topic:</strong> {source.topic}
              </p>

              {source.source_url && source.source_url !== "Unknown" ? (
                <p>
                  <strong>URL:</strong>{" "}
                  <a href={source.source_url} target="_blank" rel="noreferrer">
                    {source.source_url}
                  </a>
                </p>
              ) : (
                <p>
                  <strong>File:</strong> {source.filename}
                </p>
              )}

              <p>
                <strong>Chunk:</strong> {source.chunk_number}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;