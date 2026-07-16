# Manuscript Submission Revision Notes

## Recommended Title

A Segment-Aware Transfer Learning Framework for Reliable Early-Stage Cost Estimation of Building Retrofit Projects

## Revised Abstract

Reliable early-stage cost estimation is essential for building retrofit planning, yet prediction remains challenging because retrofit costs are often highly skewed and high-cost projects are underrepresented in available datasets. Conventional single-stage machine learning models may achieve acceptable overall accuracy but frequently underestimate extreme-cost projects, increasing the risk of budget shortfalls and unreliable investment decisions. To address this issue, this study proposes a segment-aware transfer learning framework for building retrofit cost estimation. The framework first classifies projects into low- and high-cost segments using a support vector machine classifier and then applies segment-specific artificial neural network regressors enhanced through transfer learning from data-rich to data-scarce cost segments. The framework was evaluated using a real-world dataset of 419 retrofit projects involving nursery schools and healthcare facilities, with direct construction costs ranging from approximately USD 5,000 to USD 800,000. Results show that the proposed framework achieved an average R2 of 0.9566 and reduced MAPE and RMSE by 46% and 25%, respectively, compared with the best-performing single-stage baseline. The framework also reduced prediction errors for high-cost projects, mitigating the systematic underestimation observed in conventional models. SHAP analysis further revealed that insulation, window replacement, HVAC systems, and renewable energy measures were key cost-driving factors. The proposed framework provides a reliable and interpretable decision-support approach for early-stage retrofit cost planning under skewed and data-scarce conditions.

## Keywords

Building retrofit; Cost estimation; Transfer learning; Segment-aware modeling; Decision support

## High-Priority Revisions

1. Reframe the introduction around retrofit cost uncertainty, skewed cost distributions, high-cost underestimation, and the need for segment-aware transfer learning.
2. Use consistent terminology such as "building retrofit cost", "retrofit construction cost", and "early-stage retrofit cost estimation".
3. Correct the Results section comparison: overall RMSE is 28,750 to 21,377, whereas high-cost segment RMSE should be reported as 41,163 to 34,337 for cost greater than Q3 and 47,410 to 39,797 for cost greater than Q3 + IQR/2.
4. Replace confusing "overfunding" wording with underestimation, budget shortfalls, or cost overrun risk where appropriate.
5. Clarify data leakage controls in the methodology, including how cutoffs, scalers, GridSearchCV, and hold-out test sets were handled for each random split.
6. Reduce table and figure burden where possible, especially by shortening Table 1 or moving extended literature comparisons to supplementary material.
7. Clean incomplete references and convert citation/reference style to the target journal's required format before submission.

## DIBE Resubmission Notes

If resubmitting to Developments in the Built Environment, do not submit the identical manuscript without changes. Revise the title, abstract, introduction framing, and reviewer suggestions. In the cover letter, state that the previous submission was not sent for full peer review due to reviewer unavailability, and explain that the manuscript has been reframed around building retrofit decision support and early-stage cost-risk assessment.

