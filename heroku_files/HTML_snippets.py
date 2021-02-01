Title_html = """
    <style>
        .title h1{
          user-select: none;
          font-size: 43px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 600vw 600vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
        .reportview-container .main .block-container{
            padding-top: 3em;
        }
        body {
            background-color: white;
            background-position-y: -200px;
        }
        @media (max-width: 1800px) {
            body {
                background-position-x: -500px;
            }
        }
        .Widget.stTextArea, .Widget.stTextArea textarea {
        height: 586px;
        width: 400px;
        }
        h1{
            color: black
        }
        h2{
            color: black
        }
        p{
            color: black
        }
        .sidebar-content {
            width:25rem !important;
        }
        .Widget.stTextArea, .Widget.stTextArea textarea{
        }
        .sidebar.--collapsed .sidebar-content {
         margin-left: -25rem;
        }
        .streamlit-button.small-button {
        padding: .5rem 9.8rem;
        }
        .streamlit-button.primary-button {
        background-color: white;
        }
    </style> 
    
    <div>
        <h1>Welcome to the COVID Prediction App!</h1>
    </div>
    """