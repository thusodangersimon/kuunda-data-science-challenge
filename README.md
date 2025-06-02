# Starting model
This is for the Kunnda data science challenge. Still a bit of work to do, to get 
everything working, but the basic structure is there.

# How to run
## getting the data
Copy the .Sample.env to .env and fill in the values for the AWS credentials.

## starting the model
use the following command to run the model:
```bash
docker compose up inference
```
This will start the inference server, which will listen for requests on port 8000. 
You can then send requests to the server using the following command:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{loan_csv}'
```

# to spin up the jupyter notebook
```bash
docker compose up eda
```
The server will be available at http://localhost:8890. You can then open the notebook and run the code.

# Task 1
## Describe and implement a (back) testing strategy to estimate model performance in production.
The target i used for the model is to treat each loan as independatn from eachother. This means
that the data can be split into train, development (dev) and validations sets. But even though
i spiilt the data like this, i would recomed to user a time based split, where the data is split
by date. This will give a beter estimation of how the model will perform in now verses aggregated over the
training window

# Task 2
I did not have enough time to implement much of this part. But I would have the model be served by using
fastapi, which would allow for easy scaling and deployment. Then the api would be similar and accessable to input from
outside users. If the model is deployed in on premise, then the model could be encrypted and still
produce a standart set of inputs and outputs. If deployed in the cloud, then some event/batch service
could trgger the model in the same way. 

FastAPI also alows to define swagger documentation, which can be used to docuemnt the model for training and
inference. Then the expected schema can be defined and found as part of the model.

# Task 3
## What is the most critical business consideration when evaluating model performance at Kuunda? Explain with examples.
To maximize loan portfolio growth while managing risk, the key is to strike a balance: we need a model that's 
excellent at identifying users who will repay their loans, and agents must be able to effectively utilize this product.

Instead of focusing solely on perfect discrimination, our primary metric shifts to expected risk, specifically 
the default rate of a loan relative to its amount. This allows us to strategically onboard slightly riskier new 
agents by offsetting their potential defaults with high-repaying, existing users. By doing so, we can maintain a 
robust funnel of new agents without significantly increasing the overall portfolio risk.
## If you had to explain this model to a bank or regulator, how would you communicate the risks and benefits?
When speaking to a bank, we'd focus on how our model directly boosts their bottom line and streamlines operations:

We'd start by defining the key business metrics the bank wants to optimize, such as profitability per loan, 
portfolio growth, and capital efficiency. Then, we'd demonstrate how our model accurately predicts the inherent 
risk of each loan. We'd use clear charts to illustrate various risk levels and corresponding loan amounts, showing 
their expected returns. Finally, we'd detail our robust monitoring system that proactively detects model drift, 
ensuring the model stays accurate and effective as market conditions evolve.

When speaking to a regulator, our emphasis shifts to compliance, transparency, and consumer protection:

We'd explain precisely how the model informs loan decisions, ensuring consistency and fairness. We'd then dive into
our rigorous process for monitoring the model for any signs of bias or unfairness, including regular audits and
performance reviews across different demographic segments. We'd also outline our comprehensive testing and validation
framework, proving the model's reliability and robustness. Lastly, we'd describe our clear strategy for updating and 
maintaining the model over time, ensuring it remains accurate, fair, and compliant with evolving regulations.
## How would you monitor this model in production? What data would you track and how would you trigger retraining?
### Monitoring Data and Model Performance
Keeping Tabs on Our Data:

We constantly compare the current incoming data with the data the model was trained on. Think of it like checking 
if the new questions we're asking the model are similar to the ones it studied for its exam.
To do this, we use statistical tools like the K-S test and KL-divergence. These help us see if the distributions of 
our features are shifting significantly. If they are, it's a red flag.

Tracking the Model's Default Rate:

We also closely watch the model's predicted default rate and compare it to the actual default rate we observe
in the real world.
If there's a big difference between what the model predicts and what's actually happening, then we should have an alert.

### Triggering Retraining
To make sure we catch these issues fast, we set up an automated system:

A scheduled job regularly calculates the results of all these statistical tests and performance comparisons.
These results go into a central dashboard or table.
If any of these metrics cross a predefined threshold (meaning the data or model performance has deviated too much), 
it automatically triggers an alert and initiates the process to retrain the model. This way, our model always 
stays accurate.
## What are the ethical implications of false positives and false negatives in this credit scoring model?
When our model misclassifies a business owner's risk, it can significantly affect their ability to operate and 
grow their business.

On one hand, if an agent receives a loan they shouldn't have (a false positive), it could be a sign that their 
business is already struggling. While the loan might offer a temporary fix, it could also prevent the business 
owner from addressing underlying issues and learning to manage their finances more effectively. This could ultimately 
lead to greater instability down the line.

Conversely, if an agent is denied a loan they genuinely need (a false negative)—perhaps for extra working capital 
during a busy period—it can directly result in lost revenue for their business. They might be unable to seize 
opportunities or meet demand, effectively stifling their growth and profitability. This not only hurts the individual 
business but also has broader implications for economic activity.