# Project Brief: Win Back the Customer

IT2226 AI Innovation and Deep Learning Project  \|  Team project (4 students)  \|  Duration: 17 weeks

## 1. About this project

Your team will act as an AI solutions team engaged by an online retailer. You will study the business situation, explore and prepare the company's data, and design, build and evaluate an AI solution that helps the company's staff make better decisions.

The project assesses your ability to apply two areas of AI together:

- Supervised deep learning: training, evaluating and justifying a deep learning model on the data provided.

- Agentic AI: building an AI agent that reasons, uses tools (including your trained model) and takes actions to help staff solve real business problems.

This brief describes the client, its current situation, the problem, and what its staff wish for. It deliberately does not tell you how to solve the problem. Choosing an approach and justifying it is part of what you are assessed on.

| Item | Details |
| --- | --- |
| Team size | 4 students |
| Duration | 16 weeks |
| Client (fictional) | ShopSphere Global, an online retailer |
| Data | ShopSphere data pack: 17 data tables and 4 policy documents (see section 7) |

## 2. Client background

ShopSphere Global (a fictional company created for this project) is an online retailer founded in 2022. It sells about 400 products in eight categories (Electronics, Fashion, Home & Living, Beauty, Sports & Outdoors, Toys & Kids, Groceries, and Books & Media) to customers in the United States, the United Kingdom, Germany, France and India, through its website and mobile app.

The company has grown quickly. Revenue is strongly seasonal: November (Singles' Day and Black Friday) and December are its busiest months. Its customers belong to a four-tier loyalty programme, ShopSphere Rewards (Bronze, Silver, Gold and Platinum).

## 3. The current situation

ShopSphere's leadership team is worried. The business keeps signing up new customers, but about one in three customers who have signed up have not visited the site at all in the last 90 days. Marketing spend keeps rising, while nobody can say with confidence which spending is working. The teaching team interviewed staff across the company, and their comments are summarised below.

### 3.1 CRM and Retention team

Every Monday, two CRM executives build the week's "win-back" list. They export data from the customer database, the order system, the preference centre and the helpdesk into spreadsheets, match them by hand, and choose customers mainly by how long it has been since their last order. Most customers on the list receive the same 10% voucher. Building the list takes about a day and a half.

> What staff say
> "By the time the list is ready, it's already out of date. And we keep giving vouchers to people who would have bought anyway." (CRM Manager)
> "Last quarter we sent a 20% voucher to Bronze customers by mistake. The policy says Platinum only. Nobody noticed for weeks." (CRM Executive)

### 3.2 Marketing

Marketing runs Email, Search, Display, Social and Affiliate campaigns, each targeted at a region. Results are reported in a monthly slide deck put together from the marketing platform's exports. The team cannot easily compare channels, regions or seasons.

> What staff say
> "I need to see, at a glance, what's trending and which campaigns pay back. Right now I find out a month later." (Head of Marketing)

### 3.3 Customer Service

About 30 agents handle email, live chat, phone and WhatsApp tickets. Volumes roughly double in November and December. To reply properly, agents have to look up the customer's orders, deliveries and earlier tickets in three different systems.

> What staff say
> "A customer whose parcel was lost got a 'We miss you, here's 10% off' email the same day they complained. That's how we lose people." (Customer Service Lead)

### 3.4 Operations

ShopSphere ships through seven carriers. Late deliveries are tracked, but the information isn't linked to customer behaviour.

> What staff say
> "I suspect some carriers are costing us customers, not just money. I can't prove it." (Operations Manager)

### 3.5 Merchandising and Finance

> What staff say
> "Marketing promoted a product we had zero stock of, and discounted items with margins too thin to discount." (Merchandising Lead)
> "Each country gets a weekly retention budget. I want to know before the money is spent that it's going to the right customers, and afterwards whether it worked." (Finance Controller)

## 4. Problem statement

> Problem statement
> ShopSphere's staff make important weekly decisions (who to re-engage, how, with which offer and through which channel) without knowing which customers are likely to purchase soon and which are drifting away. The information they need is spread across separate systems and policy documents, and pulling it together by hand is slow, inconsistent and error-prone. Managers also lack a timely view of the trends that would tell them whether these decisions are working.
> How might ShopSphere use its customer, behavioural and operational data, together with AI, to predict which customers will make a purchase in the next 30 days, and help its staff turn that prediction into timely, consistent, policy-compliant and cost-effective decisions?

## 5. Stakeholder wish list

The wish list below comes from ShopSphere's stakeholders. It is a list of needs, not a specification. You must explain which items you chose to address, which you did not, and why.

| # | Wish | Raised by |
| --- | --- | --- |
| W1 | Know which customers are likely to make a purchase in the next 30 days. | CRM Manager |
| W2 | Understand why a customer is likely or unlikely to purchase, in terms staff can act on. | CRM Manager, Country Managers |
| W3 | Each week, get a recommended re-engagement list (who to contact, which offer, which channel) that stays within each country's budget. | CRM Manager, Finance Controller |
| W4 | Be certain every recommendation follows company policy: consent, contact frequency, open complaints, tier eligibility, stock and margin. | CRM Manager, Customer Service Lead |
| W5 | Before replying, give a customer service agent a one-screen summary of the customer: orders, deliveries, earlier tickets and any recent offers. | Customer Service Lead |
| W6 | A dashboard showing trends in sales, traffic, campaign performance, delivery performance and customer activity, by country and over time. | Head of Marketing, Operations Manager, Leadership |
| W7 | Ask business questions in plain language and get one answer that pulls together information from several sources. | Leadership, Finance Controller |
| W8 | Draft personalised messages in the customer's preferred language and channel. | CRM Executive |
| W9 | Keep humans in control: nothing is sent without approval, and every decision has a recorded reason. | CRM Manager |
| W10 | After a campaign, find out whether the money spent made a difference. | Finance Controller |

## 6. Project requirements

Whatever approach you choose, your solution must:

1. include a supervised deep learning model trained on the data pack to predict the target in train.csv, evaluated properly and compared fairly against at least one non-deep-learning baseline;

2. include an agentic AI component that uses your trained model as part of its reasoning or actions;

3. address at least five wish-list items, including W1, W3 and W6;

4. use information from at least four different tables or documents in the data pack when making a decision;

5. be demonstrated as a working prototype that ShopSphere staff could realistically use;

6. be evaluated with evidence: both the model and the agent;

7. consider responsible AI: fairness, privacy and consent, transparency, and the risk of the system being wrong.

## 7. The data pack

The data pack is ShopSphere's data extract as at 30 September 2024. The target in train.csv shows whether each customer made at least one purchase between 1 and 30 October 2024. All the data is synthetic and contains no real personal information.

The core tables follow the structure of the Kaggle competition Global E-Commerce Customer Purchase Prediction. They have been regenerated and extended by the NYP teaching team to reflect ShopSphere's operations. The full data dictionary is in Appendix A and in DATA_README.md in the data pack.

| Group | Files | What they describe |
| --- | --- | --- |
| Customer | customers.csv, customer_preferences.csv, geo_data.csv | Profiles, loyalty tier, consent and contact preferences, regional socio-economic data |
| Behaviour | sessions.csv, transactions.csv, transaction_items.csv | Browsing sessions, orders and order lines, January 2022 to September 2024 |
| Marketing | marketing_campaigns.csv, campaign_daily_performance.csv, retention_contact_history.csv | Campaigns (completed, active and planned), daily spend and results, retention offers sent |
| Operations | products.csv, deliveries.csv, support_tickets.csv | Catalogue, stock and margin; shipments and delays; customer service tickets with customers' messages |
| Finance and CRM | offer_catalogue.csv, retention_budget_2024Q4.csv | Allowed retention offers; weekly budget per country for Q4 2024 |
| Labels | train.csv, test.csv, sample_submission.csv | 64,000 labelled customers; 16,000 customers to predict |
| Policies | retention_offer_policy.md, shipping_and_returns_policy.md, customer_service_sla.md, loyalty_programme.md | ShopSphere's business rules |

> Watch out: The data is messy
> Like real company data, the pack contains inconsistent spellings, several encodings for true/false, numbers stored as text, missing values, outliers and duplicates. Some tables were exported by different systems. Treat data preparation as a real part of the project, and check the data itself rather than relying only on the documentation.

You may create additional synthetic data if your solution needs it. Any data you create must be documented and clearly labelled as synthetic. The data pack itself must not be altered.

## 8. Deliverables and milestones

| Week | Deliverable | What it must contain |
| --- | --- | --- |
| 3 | Project proposal (max 6 pages) and 10-minute pitch | Stakeholders and their needs; the wish-list items chosen and why; initial data exploration and data quality findings; proposed approach; plan and roles; risks. |
| 8 | Model checkpoint (short report + notebook) | Data preparation and feature pipeline, baseline(s), deep learning model(s), evaluation, a first test-set submission, and the model you intend to use, with justification. |
| 13 | Prototype demonstration (15 minutes, live) | Working end-to-end prototype, including the dashboard, demonstrated on realistic ShopSphere scenarios, with early evaluation results. |
| 16 | Final report (max 25 pages + appendices) | Problem, data, model development and evaluation, agent and dashboard design and evaluation, responsible AI, results against the wish list, limitations, future work. |
| 16 | Final presentation and live demo (20 minutes + Q&A) | Presented to a panel acting as ShopSphere leadership. Every member presents. |
| 16 | Code repository and final test-set submission | Reproducible code, notebooks and README. No API keys committed. |
| 16 | Individual reflection and peer evaluation | Each member's contribution and learning; an evaluation of teammates' contributions. |

## 9. Teamwork, constraints and rules

- Each member must own at least one major component and be able to explain it in Q&A.

- Use Git from Week 1 and keep a shared task board and weekly meeting log.

- Use only open-source libraries and the AI models or APIs your lecturer approves. Stay within any API budget you are given.

- Declare any use of generative AI tools in your report: which tools you used and what for.

- Your prototype must not connect to any real messaging, email or payment system.

## 10. Assessment

A detailed rubric will be released separately. Broadly, you will be assessed on:

- understanding of the stakeholders and the problem;

- data preparation and deep learning;

- agentic AI;

- the dashboard and analytics;

- evaluation;

- responsible AI;

- the prototype, report and presentation;

- individual contribution.

# Appendix A: Data dictionary

| File | Columns |
| --- | --- |
| customers.csv | customer_id; age (or "unknown"); country_code; region; signup_date; loyalty_score (0–1 internal engagement index); email_open_rate; discount_usage_rate (share of the customer's orders that used a discount); avg_review_score (1–5); referral_code; customer_tier (based on 12-month spend at the cutoff); credit_limit; noise_customer_0–4 (system fields with no business meaning) |
| sessions.csv | session_id; customer_id; session_timestamp; session_duration (s); pages_viewed; cart_additions; bounce_flag; traffic_source; device_type; campaign_id (blank if not from a campaign); geo_ip_region |
| transactions.csv | transaction_id; customer_id; transaction_timestamp; order_value (after discount, including shipping); items_count; payment_method; discount_applied; shipping_speed; high_value_flag (above $250); noise_trans_0–1 |
| transaction_items.csv | transaction_id; line_no; product_id; quantity; unit_price (list price) |
| products.csv | product_id; category; product_name; unit_price; margin_pct; stock_on_hand; next_restock_date; is_promotable |
| deliveries.csv | transaction_id; carrier; shipping_speed; ship_date; promised_date; delivered_date; delay_days; delivery_status (Delivered / Delivered Late / Lost / Returned / In Transit) |
| support_tickets.csv | ticket_id; customer_id; transaction_id; created_date; channel; category; priority; status (at the cutoff); first_response_hours; resolution_hours; csat_score (1–5); message (the customer's first message) |
| customer_preferences.csv | customer_id; email_opt_in; sms_opt_in; preferred_channel; preferred_language; do_not_contact; last_updated |
| marketing_campaigns.csv | campaign_id; campaign_type; campaign_budget; region_target; start_date; end_date; status (Completed / Active / Planned) |
| campaign_daily_performance.csv | campaign_id; date; spend; impressions; clicks; attributed_sessions; attributed_orders; attributed_revenue (same-day attribution) |
| retention_contact_history.csv | contact_id; customer_id; contact_date; channel; offer_code; purchased_within_14d; redeemed; redemption_date (July–September 2024) |
| offer_catalogue.csv | offer_code; description; offer_type; value; eligible_tiers; counts_toward_monthly_limit; requires_promotable_items |
| retention_budget_2024Q4.csv | week_start; country; retention_budget_usd |
| geo_data.csv | geo_ip_region; average_income; urban_ratio; internet_penetration; region_tier |
| train.csv / test.csv | customer_id; target (train only: 1 = at least one purchase between 1 and 30 October 2024) |

My Additional Notes:
It is very important to look at the people who raised each wish to truely see what is the problem since the wishes may not be specific.

W1 — Predict purchases within 30 days: Develop a supervised deep learning model that estimates each customer's likelihood of purchasing. 
W3 — Weekly re-engagement recommendations within country budgets: Use purchase predictions, customer information, eligible offers and country budgets to generate a prioritised list of customers to contact quickly so that it is not out of date. This supports more targeted and cost-effective retention campaigns.This helps CRM staff contact customers who are unlikely to order rather than “keep giving vouchers to people who would have bought anyway.” And also using user data to indicate the type of voucher to send. We could also go further to have 
W6 — Business analytics dashboard: Display sales, traffic, campaign performance, delivery performance and customer activity by country and over time. This helps Marketing, Operations and Leadership monitor trends and make timely decisions. This also can help find if a carrier is failing at their duty by tracking performance by company.
W9 — Human approval and recorded reasons: Require staff approval before any simulated outreach and record recommendation reasons and approval decisions. This maintains human control, improves accountability and reduces the risk of inappropriate automated actions.
W10 — Evaluate campaign effectiveness: Track campaign spending and outcomes, such as purchases, redemptions and attributed revenue. Where suitable comparison data exists, estimate incremental impact to help Finance assess whether campaigns make a difference.

This Project only needs to be run locally so all operations should reflect that.