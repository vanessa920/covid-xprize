possible issues to look out for:
- plotly, PIL etc not included in `requirements.txt` update it...
- looks like old import requirements were left over from an NLP project
- need to add population somehow so the predictions are a real number, not: per 100k values
- get rid of "show analysis by country" checkbox and just add a "None" entry to the dropdown list which is selected by default and shows the "Country level analysis" section with a prompt saying to select a country from the dropdown
- put in a one/two sentence explanation of what the daily new cases plot means
- fix graph so date doesn't include time