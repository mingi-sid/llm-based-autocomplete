import './App.css';
import React, { Children, useCallback, useState } from 'react'
import { createEditor, Editor, Transforms, Element } from 'slate'
import { Slate, Editable, withReact, DefaultElement } from 'slate-react'

const AUTOCOMPLETE_DELAY = 1000;

const initialValue = [
  {
    type: 'paragraph',
    children: [{ text: 'A line of text in a paragraph.'},
    ],
  }
]


function PreviewElement( props ) {
  return (
    <p {...props.attributes}
      style={{color: 'gray'}}>
      {props.children}
    </p>
  )
}

function App() {
  const [editor] = useState(() => withReact(createEditor()));
  let autocompleteTimer;

  const renderElement = useCallback( (props) => {
    switch (props.element.type) {
      case 'preview':
        return <PreviewElement {...props} />
      default:
        return <DefaultElement {...props} />
    }
  }, [])

  function Leaf(props) {
    return (
      <span
        {...props.attributes}
        style={{ color: (props.leaf.preview ? 'gray' : 'inherit') }}
      >
        {props.children}
      </span>
    )
  }
  const renderLeaf = useCallback((props) => {
    return <Leaf {...props} />
  }, [])
  

  function callAutocomplete() {
    console.log('Called autocomplete!');
    const previewText = 'Called autocomplete!\nCalled autocomplete!'
    const previewTextList = previewText.split('\n')
    console.log(editor.selection);
    const startPoint = editor.selection.focus;
    Editor.insertText(
      editor,
      previewTextList[0]
    )
    Transforms.insertNodes(
      editor,
      {
        type: 'preview',
        children: [{ text: previewTextList[1] }],
      }
    );
    Transforms.select(editor, {
      anchor: startPoint,
      focus: editor.selection.focus,
    })
    Editor.addMark(editor, 'preview')
  }

  return (
    <Slate editor={editor} initialValue={initialValue}>
      <Editable 
        renderElement={renderElement}
        renderLeaf={renderLeaf}
        onKeyDown={ (e) => {
          if(e.key === " " || e.key === "Enter") {
            clearTimeout(autocompleteTimer);
            autocompleteTimer = setTimeout(callAutocomplete, AUTOCOMPLETE_DELAY);
          }
          else {
            clearTimeout(autocompleteTimer);
          }
        } }
      />
    </Slate>
  )
}

export default App;
