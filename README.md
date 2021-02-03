# ROUND 2:

## Website Notes:

We need to decide how we'll handle the costs for the website. Uniform costs will likeley be the best baseline since it won't end up with messed up NPI preferences. (NECESSARY FOR MVP)

There should be a slider for the stringency (user chooses preference for less case #s or less NPIs) which will select one of the 10 outputs from the prescriptor for that region.

Since some countries can't implement certain NPIs or it doesn't make sense there should be checkboxes for users to decide which NPIs they can omit from the model. (NOT NECESSARY FOR MVP, DEFINITE 2ND ITERATION)

See [evolution.ml]('https://evolution.ml/demos/npidashboard/') which the contest judges provided for us as an example.

## Model Functionality:

We will use the random costs provided by the judges for our model to make sure that the model can run on a wide range of input values. The judges specified that we don't need to focus on coming up with costs for this round. Be sure to mention HOW we WOULD come up with them after this round in the qualitative submission.

The stringency should automatically be determined in the program to find the best tradeoff of: low cost : low case #s
^^ this part still needs to be figured out

### Inputs are:
* Coefficients from previous round for each country's NPI effectiveness
* Cost for each country's NPI
^^ these will be used to make a custom Effectiveness/cost coefficient which will be input to the model
* Oxford daily case data

### Next step will be the engineered feature:
* (Effectiveness * stringency factor) / (intervention cost)
The stringency factor will be:
    * automatacilly determined for the sandbox version
    * set to the middle value on a user adjustable slider for the web version

### Prediction:
Whatever model we choose will take in the Oxford data AND the engineered feature to determine:
    * initial NPIs for each GEOID
    * rate that we should stop enforcing the NPIs over the next 30/60/90 days

![prescription_model.png]('prescription_model.png')
