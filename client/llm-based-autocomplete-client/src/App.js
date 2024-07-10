
/* LLM Based Autocomplete Client

This is the client side of the LLM Based Autocomplete project.
It is a React app that will send a request to the server to get the autocomplete suggestions for the input text,
when (1) the user presses the enter key or (2) the user pauses typing for 1 second.
The server will then send back the auto-completed paragraph to the client, which will display them in another text box.
The client will also display the number of characters in the input text. */
import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  // State variables
  const [inputText, setInputText] = useState("");
  const [numCharacters, setNumCharacters] = useState(0);
  const [autocompleteSuggestions, setAutocompleteSuggestions] = useState("");
  const [renderedText, setRenderedText] = useState("");

  // Event handlers
  const handleInputTextChange = (event) => {
    setInputText(event.target.value);
  };

  const handleNumCharactersChange = (event) => {
    setNumCharacters(event.target.value);
  };

  const getAutocompleteSuggestions = async () => {
    try {
      const response = await fetch('http://localhost:5000/autocomplete', {
        method: 'POST',
        //mode: 'no-cors', // Set the mode to 'no-cors' to disable CORS
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ inputText: inputText }),
      });
      const body = await response.json();
      console.log(body);
      setAutocompleteSuggestions(body.autoCompletedText);
    } catch (error) {
      console.error(error);
    };
  };

  // Effect hooks
  useEffect(() => {
    // Add an event listener to detect when the user presses the enter key
    document.addEventListener("keydown", handleKeyDown);

    // On component unmount, remove the event listener
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      getAutocompleteSuggestions();
    }
  };

  useEffect(() => {
    // Add an event listener to detect when the user pauses typing for 1 second
    const timer = setTimeout(() => {
      getAutocompleteSuggestions();
    }, 1000);

    // On component unmount, remove the event listener
    return () => {
      clearTimeout(timer);
    };
  }, [inputText]);

  useEffect(() => {
    setNumCharacters(inputText.length);
  }, [inputText]);

/*   function renderText() {
    // Split by \n
    const inputTextLines = inputText.split('\n');
    const autocompleteSuggestionsLines = autocompleteSuggestions.split('\n');
    // Add <br /> to the end of each line and concatenate them
    let renderedText = "";
    for (let i = 0; i < inputTextLines.length; i++) {
      renderedText += inputTextLines[i] + "<br />";
      if (i < autocompleteSuggestionsLines.length) {
        renderedText += autocompleteSuggestionsLines[i] + "<br />";
      }
    }
    return renderedText;
  };

  useEffect(() => {
    setRenderedText(renderText());
  }, [inputText, autocompleteSuggestions]); */

  return (
    <div className="App">
      <h1>LLM Based Autocomplete Client</h1>

      <div className="autocomplete">
        <label>Input Text:</label>
        <br />
        <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center' }}>
          <div id="input-container">
            <textarea
              id="input-text"
              name="input-text"
              value={inputText}
              onChange={handleInputTextChange}
            />
            <br />
            <label>Number of Characters: {numCharacters}</label>
          </div>
          <div id="suggestion-container">
            <div
              className='autocomplete-suggestion'
            >
              {inputText} <span style={{color: 'gray'}}> {autocompleteSuggestions} </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
