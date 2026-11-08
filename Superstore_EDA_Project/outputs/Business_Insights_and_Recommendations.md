# Business Insights and Recommendations

*Superstore Retail Sales EDA. All figures are calculated from the uploaded dataset (cleaned: 9,993 order lines, 2015-01-03 to 2018-12-30).*

## Key insights

### 1. Discounts above 20% are almost always loss-making

- **Observation:** Order lines discounted by more than 20% rarely make money, while undiscounted lines are very profitable.
- **Evidence:** 1,392 lines (13.9% of all lines) are discounted above 20%. Together they lost \$135.4K, and 97% of them individually lost money. Lines with no discount earn a 29.5% margin versus 11.9% for 1-20% discounts. The Spearman correlation between discount and profit is -0.54.
- **Business meaning:** Deep discounts and profit move in opposite directions, and every discount band above 20% has a negative total profit. This is an association, not proof of cause, because discounted lines also differ in product mix and region.
- **Recommended action:** Set a 20% standard discount ceiling, require manager approval above it, and track margin after discount every month. Pilot the rule in one region first to measure the real effect.

### 2. Furniture sells a lot but earns very little

- **Observation:** Furniture is about a third of sales, but contributes only a small part of profit.
- **Evidence:** Furniture is 32.3% of sales but only 6.4% of profit (margin 2.5% vs 17.4% for Technology and 17.0% for Office Supplies). Tables lost \$17.7K (-8.6% margin) and Bookcases lost \$3.5K. Table lines discounted above 20% lost \$30.7K, while Table lines at 20% or less made \$13.0K.
- **Business meaning:** The category's weak profit is concentrated in two sub-categories, and heavy discounting appears to be a major part of the Tables problem. Chairs and Furnishings remain profitable.
- **Recommended action:** Review prices and supplier costs for Tables and Bookcases, stop deep discounts on them, and move promotional budget toward Technology and Office Supplies.

### 3. Sales are concentrated in September to December

- **Observation:** Demand is strongly seasonal, with a repeating year-end peak and a weak start to the year.
- **Evidence:** September to December generate 51.6% of sales (November alone 15.3%). February is the weakest month (2.6% of sales). September, November and December were the three biggest months in 4 of 4 years.
- **Business meaning:** The pattern is stable, so inventory, staffing and marketing can be planned in advance. The likely cause is year-end buying, although the data cannot confirm the reason.
- **Recommended action:** Build stock and shipping capacity before September. In January-February use bundles and value offers rather than deep discounts to lift the slow months.

### 4. The Central region has the weakest margin

- **Observation:** West is the best region on sales and profit; Central earns far less per dollar of sales.
- **Evidence:** Central has \$501.2K in sales (21.8%) but a 7.9% margin versus 14.9% in West. Its average discount is 24.0% versus 10.9% in West, and 27.7% of Central lines are discounted above 20% (West: 3.7%). Texas lost \$25.7K with an average discount of 37.0%. Inside Central, Indiana (average discount 0.0%) earned a 34.3% margin, while Illinois (average discount 39.0%) ran at -15.7%.
- **Business meaning:** Central's lower margin coincides with much heavier discounting. Product mix and local costs may also contribute, so a drill-down is needed before drawing a final conclusion.
- **Recommended action:** Audit discount approvals in Central, starting with Texas. Compare Central and West discounting product by product, and set a minimum margin target for each region.

### 5. Sales growth is strong, but margin slipped in 2018

- **Observation:** Revenue is growing faster than profit.
- **Evidence:** Sales grew 29.5% in 2017 and 20.4% in 2018 (to \$733.2K), but 2018 profit grew only 14.2% and the margin fell from 13.4% to 12.7%.
- **Business meaning:** Each extra dollar of sales is now worth slightly less profit, so growth is becoming a little less efficient.
- **Recommended action:** Make profit margin a headline KPI next to sales, and link sales targets to margin, not revenue alone.

### 6. The customer base is broad, and the biggest customer by sales lost money

- **Observation:** Revenue is not dependent on a few accounts, but sales rank hides unprofitable customers.
- **Evidence:** There are 793 customers. The top 10 contribute only 6.7% of sales and the top 20% contribute 48.1%. The #1 customer by sales (Sean Miller) generated \$25.0K in sales but -\$2.0K in profit. 155 customers (19.5%) have negative total profit, adding up to -\$71.2K.
- **Business meaning:** Concentration risk is low, but ranking customers by sales alone can reward accounts that cost money.
- **Recommended action:** Show profit next to sales in customer reports, review the discounts given to loss-making accounts, and use loyalty offers to grow the broad middle tier.

### 7. Returned orders take a noticeable share of sales

- **Observation:** A measurable part of orders is returned, and returned orders are different in size.
- **Evidence:** 296 orders (5.9% of all orders) appear in the appended Returns table, worth \$180.5K (7.9% of sales). Their average order value is \$610 versus \$449 for other orders. Technology has the highest share of its sales returned (8.7%).
- **Business meaning:** Returned orders are worth more than a typical order, so each return costs more than the order count suggests. Return reasons and refund amounts are not in the data, so the exact profit impact cannot be measured here.
- **Recommended action:** Start recording return reasons and refund values, and investigate Technology returns first.

## Recommendations (in priority order)

**Do now (policy changes, low cost)**
1. **Introduce discount guardrails.** Cap standard discounts at 20% and require approval above that (Insight 1).
2. **Audit the Central region**, starting with Texas, and compare its discounting with West (Insight 4).

**Do next (1-2 quarters)**
3. **Reprice or rework Tables and Bookcases**: check supplier costs, stop deep discounts, shift promotion to Technology and Office Supplies (Insight 2).
4. **Plan around the seasonal peak**: stock and staff up before September; use value bundles, not deep discounts, to lift January-February (Insight 3).

**Build into regular reporting**
5. **Report profit margin and customer profit, not only sales** (Insights 5 and 6).
6. **Capture return reasons and refund values** so returns can be linked to cost (Insight 7).
7. **Fix the data export.** Export Sales, People and Returns as separate files.


## Limitations
- Discount and profit are associated; causation is not proven (product mix and region also differ).
- No cost, return-reason or refund data; period covers 2015-2018 only.
- 11 missing Postal Codes (Burlington, VT) were filled with 05401; no metric uses Postal Code.
- Returns were taken from a table appended to the sales file; every listed Order ID was treated as returned.