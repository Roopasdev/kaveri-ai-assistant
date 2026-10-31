# Kaveri Exports sales database: data dictionary

## regions

| Column | Meaning |
|---|---|
| `region_id` | Primary key, R1 to R5. |
| `region_name` | Europe, Middle East, Asia Pacific, Americas or South Asia. |
| `sales_head` | The person responsible for sales in the region. |
| `target_2024` | Revenue target for 2024 in USD. |

## customers

| Column | Meaning |
|---|---|
| `customer_id` | Primary key, C001 to C128. |
| `company_name` | The customer's company. |
| `country` | Country the customer is based in. |
| `city` | City the customer is based in. |
| `region_id` | Foreign key to regions.region_id. |
| `segment` | Wholesale, Retail or Distributor. |
| `joined_on` | Date the customer was onboarded (YYYY-MM-DD). |

## products

| Column | Meaning |
|---|---|
| `product_id` | Primary key, P01 onwards. |
| `product_name` | Product name. |
| `category` | Spices, Tea & Coffee, Textiles, Coir & Natural, Handicrafts, Leather or Food. |
| `list_price` | Catalogue price per unit in USD. |

## orders

| Column | Meaning |
|---|---|
| `order_id` | Primary key, O00001 onwards, in date order. |
| `customer_id` | Foreign key to customers.customer_id. |
| `order_date` | Date of the order (YYYY-MM-DD). |
| `status` | delivered, shipped or cancelled. |
| `sales_channel` | Direct, Distributor or Online. |
| `total_amount` | Order value in USD. 0.00 for cancelled orders. Revenue = SUM(total_amount). |

## order_items

| Column | Meaning |
|---|---|
| `order_item_id` | Primary key. |
| `order_id` | Foreign key to orders.order_id. |
| `product_id` | Foreign key to products.product_id. |
| `quantity` | Units ordered. |
| `unit_price` | Price per unit actually charged in USD (list price less any discount). |
| `line_total` | quantity * unit_price. Kept for cancelled orders too, for audit. |

## Definitions

- **Revenue** = `SUM(orders.total_amount)`. Cancelled orders have `total_amount = 0.00`,
  so they never add to revenue.
- **A year's revenue** uses `order_date` in that year.
- **An order's country** is its customer's country.
