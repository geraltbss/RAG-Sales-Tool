import pandas as pd
import numpy as np
import json
from pathlib import Path


def load_dataset(filepath: str = "Sample - Superstore.csv") -> pd.DataFrame:
    # Load and clean the Superstore dataset.
    df = pd.read_csv(filepath, encoding="latin-1")

    # Parse dates
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="mixed")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="mixed")

    # Add time-based columns for aggregation
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    df["YearQuarter"] = df["Order Date"].dt.to_period("Q").astype(str)

    # Compute profit margin
    df["Profit Margin"] = np.where(df["Sales"] > 0, df["Profit"] / df["Sales"] * 100, 0)

    print(f"Loaded {len(df)} rows, date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
    print(f"Categories: {df['Category'].unique().tolist()}")
    print(f"Regions: {df['Region'].unique().tolist()}")
    return df

# 1. Transaction Descriptions (grouped by Order ID)

def create_transaction_descriptions(df: pd.DataFrame) -> list[dict]:
    # Group transactions by Order ID and create natural language descriptions.
    docs = []

    for order_id, group in df.groupby("Order ID"):
        first = group.iloc[0]
        items = []
        total_sales = 0
        total_profit = 0
        total_qty = 0
        categories = set()

        for _, row in group.iterrows():
            items.append(
                f"  - {row['Product Name']} ({row['Sub-Category']}, {row['Category']}): "
                f"${row['Sales']:.2f}, qty {row['Quantity']}, "
                f"discount {row['Discount']*100:.0f}%, profit ${row['Profit']:.2f}"
            )
            total_sales += row["Sales"]
            total_profit += row["Profit"]
            total_qty += row["Quantity"]
            categories.add(row["Category"])

        margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

        text = (
            f"Order {order_id} placed on {first['Order Date'].strftime('%B %d, %Y')} "
            f"by {first['Customer Name']} ({first['Segment']} segment) "
            f"from {first['City']}, {first['State']} ({first['Region']} region). "
            f"Shipped via {first['Ship Mode']}.\n"
            f"Items ({len(group)}):\n" + "\n".join(items) + "\n"
            f"Order total: ${total_sales:.2f}, total profit: ${total_profit:.2f}, "
            f"profit margin: {margin:.1f}%, total quantity: {total_qty}."
        )

        docs.append({
            "text": text,
            "metadata": {
                "type": "transaction",
                "order_id": order_id,
                "date": first["Order Date"].strftime("%Y-%m-%d"),
                "year": int(first["Year"]),
                "month": int(first["Month"]),
                "quarter": int(first["Quarter"]),
                "region": first["Region"],
                "state": first["State"],
                "city": first["City"],
                "segment": first["Segment"],
                "categories": sorted(categories),
                "total_sales": round(total_sales, 2),
                "total_profit": round(total_profit, 2),
            },
        })

    print(f"Created {len(docs)} transaction descriptions")
    return docs

# 2. Aggregated Summaries

def _fmt(val: float) -> str:
    return f"${val:,.2f}"


def create_monthly_summaries(df: pd.DataFrame) -> list[dict]:
    #Monthly sales/profit summaries.
    docs = []
    grouped = df.groupby("YearMonth").agg(
        sales=("Sales", "sum"),
        profit=("Profit", "sum"),
        orders=("Order ID", "nunique"),
        quantity=("Quantity", "sum"),
        avg_discount=("Discount", "mean"),
    ).reset_index()

    for _, row in grouped.iterrows():
        margin = (row["profit"] / row["sales"] * 100) if row["sales"] > 0 else 0
        text = (
            f"Monthly summary for {row['YearMonth']}: "
            f"Total sales {_fmt(row['sales'])}, total profit {_fmt(row['profit'])}, "
            f"profit margin {margin:.1f}%. "
            f"{int(row['orders'])} unique orders, {int(row['quantity'])} items sold, "
            f"average discount {row['avg_discount']*100:.1f}%."
        )
        year, month = row["YearMonth"].split("-")
        docs.append({
            "text": text,
            "metadata": {
                "type": "monthly_summary",
                "year": int(year),
                "month": int(month),
                "period": row["YearMonth"],
                "total_sales": round(row["sales"], 2),
                "total_profit": round(row["profit"], 2),
            },
        })
    print(f"Created {len(docs)} monthly summaries")
    return docs


def create_quarterly_summaries(df: pd.DataFrame) -> list[dict]:
    # Quarterly sales/profit summaries.
    docs = []
    grouped = df.groupby("YearQuarter").agg(
        sales=("Sales", "sum"),
        profit=("Profit", "sum"),
        orders=("Order ID", "nunique"),
        quantity=("Quantity", "sum"),
    ).reset_index()

    for _, row in grouped.iterrows():
        margin = (row["profit"] / row["sales"] * 100) if row["sales"] > 0 else 0
        text = (
            f"Quarterly summary for {row['YearQuarter']}: "
            f"Total sales {_fmt(row['sales'])}, total profit {_fmt(row['profit'])}, "
            f"profit margin {margin:.1f}%. "
            f"{int(row['orders'])} orders, {int(row['quantity'])} items sold."
        )
        docs.append({
            "text": text,
            "metadata": {
                "type": "quarterly_summary",
                "period": row["YearQuarter"],
                "total_sales": round(row["sales"], 2),
                "total_profit": round(row["profit"], 2),
            },
        })
    print(f"Created {len(docs)} quarterly summaries")
    return docs


def create_yearly_summaries(df: pd.DataFrame) -> list[dict]:
    # Yearly sales/profit summaries with year-over-year comparison.
    docs = []
    grouped = df.groupby("Year").agg(
        sales=("Sales", "sum"),
        profit=("Profit", "sum"),
        orders=("Order ID", "nunique"),
        quantity=("Quantity", "sum"),
        avg_discount=("Discount", "mean"),
    ).reset_index().sort_values("Year")

    prev_sales = None
    for _, row in grouped.iterrows():
        margin = (row["profit"] / row["sales"] * 100) if row["sales"] > 0 else 0
        yoy = ""
        if prev_sales is not None:
            change = (row["sales"] - prev_sales) / prev_sales * 100
            yoy = f" Year-over-year sales change: {change:+.1f}%."
        text = (
            f"Yearly summary for {int(row['Year'])}: "
            f"Total sales {_fmt(row['sales'])}, total profit {_fmt(row['profit'])}, "
            f"profit margin {margin:.1f}%. "
            f"{int(row['orders'])} orders, {int(row['quantity'])} items sold, "
            f"average discount {row['avg_discount']*100:.1f}%.{yoy}"
        )
        docs.append({
            "text": text,
            "metadata": {
                "type": "yearly_summary",
                "year": int(row["Year"]),
                "total_sales": round(row["sales"], 2),
                "total_profit": round(row["profit"], 2),
            },
        })
        prev_sales = row["sales"]
    print(f"Created {len(docs)} yearly summaries")
    return docs


def create_category_summaries(df: pd.DataFrame) -> list[dict]:
    # Per-category and per-subcategory summaries.
    docs = []

    # Category level
    for cat, group in df.groupby("Category"):
        sales = group["Sales"].sum()
        profit = group["Profit"].sum()
        margin = (profit / sales * 100) if sales > 0 else 0
        top_sub = group.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)

        text = (
            f"Category summary for {cat}: "
            f"Total sales {_fmt(sales)}, total profit {_fmt(profit)}, "
            f"profit margin {margin:.1f}%. "
            f"{group['Order ID'].nunique()} orders, {group['Quantity'].sum()} items sold. "
            f"Sub-categories ranked by sales: "
            + ", ".join(f"{sub} ({_fmt(val)})" for sub, val in top_sub.items())
            + "."
        )
        docs.append({
            "text": text,
            "metadata": {"type": "category_summary", "category": cat},
        })

    # Sub-category level
    for (cat, sub), group in df.groupby(["Category", "Sub-Category"]):
        sales = group["Sales"].sum()
        profit = group["Profit"].sum()
        margin = (profit / sales * 100) if sales > 0 else 0
        avg_disc = group["Discount"].mean() * 100

        text = (
            f"Sub-category summary for {sub} (under {cat}): "
            f"Total sales {_fmt(sales)}, total profit {_fmt(profit)}, "
            f"profit margin {margin:.1f}%. "
            f"{len(group)} transactions, average discount {avg_disc:.1f}%."
        )
        docs.append({
            "text": text,
            "metadata": {"type": "subcategory_summary", "category": cat, "subcategory": sub},
        })

    print(f"Created {len(docs)} category/subcategory summaries")
    return docs


def create_regional_summaries(df: pd.DataFrame) -> list[dict]:
    # Per-region and per-state summaries.
    docs = []

    # Region level
    for region, group in df.groupby("Region"):
        sales = group["Sales"].sum()
        profit = group["Profit"].sum()
        margin = (profit / sales * 100) if sales > 0 else 0
        top_states = group.groupby("State")["Sales"].sum().sort_values(ascending=False).head(5)

        text = (
            f"Regional summary for {region}: "
            f"Total sales {_fmt(sales)}, total profit {_fmt(profit)}, "
            f"profit margin {margin:.1f}%. "
            f"{group['Order ID'].nunique()} orders. "
            f"Top states by sales: "
            + ", ".join(f"{st} ({_fmt(val)})" for st, val in top_states.items())
            + "."
        )
        docs.append({
            "text": text,
            "metadata": {"type": "regional_summary", "region": region},
        })

    # State level
    for state, group in df.groupby("State"):
        sales = group["Sales"].sum()
        profit = group["Profit"].sum()
        margin = (profit / sales * 100) if sales > 0 else 0
        region = group["Region"].iloc[0]
        top_cities = group.groupby("City")["Sales"].sum().sort_values(ascending=False).head(3)

        text = (
            f"State summary for {state} ({region} region): "
            f"Total sales {_fmt(sales)}, total profit {_fmt(profit)}, "
            f"profit margin {margin:.1f}%. "
            f"{group['Order ID'].nunique()} orders. "
            f"Top cities: "
            + ", ".join(f"{c} ({_fmt(v)})" for c, v in top_cities.items())
            + "."
        )
        docs.append({
            "text": text,
            "metadata": {"type": "state_summary", "state": state, "region": region},
        })

    print(f"Created {len(docs)} regional/state summaries")
    return docs

# 3. Statistical Summaries

def create_statistical_summaries(df: pd.DataFrame) -> list[dict]:
    # Top/bottom performers, discount analysis, trend insights.
    docs = []

    # Overall dataset summary
    text = (
        f"Dataset overview: {len(df)} transactions from "
        f"{df['Order Date'].min().strftime('%B %Y')} to {df['Order Date'].max().strftime('%B %Y')}. "
        f"Total sales: {_fmt(df['Sales'].sum())}, total profit: {_fmt(df['Profit'].sum())}. "
        f"{df['Order ID'].nunique()} unique orders, {df['Customer ID'].nunique()} unique customers. "
        f"3 categories: {', '.join(df['Category'].unique())}. "
        f"4 regions: {', '.join(df['Region'].unique())}. "
        f"Segments: {', '.join(df['Segment'].unique())}."
    )
    docs.append({"text": text, "metadata": {"type": "statistical", "topic": "overview"}})

    # Top 10 products by sales
    top_products = df.groupby("Product Name").agg(
        sales=("Sales", "sum"), profit=("Profit", "sum"), qty=("Quantity", "sum")
    ).sort_values("sales", ascending=False).head(10)

    lines = [f"Top 10 products by total sales:"]
    for i, (name, row) in enumerate(top_products.iterrows(), 1):
        lines.append(f"  {i}. {name}: sales {_fmt(row['sales'])}, profit {_fmt(row['profit'])}, qty {int(row['qty'])}")
    docs.append({"text": "\n".join(lines), "metadata": {"type": "statistical", "topic": "top_products_sales"}})

    # Top 10 products by profit
    top_profit = df.groupby("Product Name").agg(
        sales=("Sales", "sum"), profit=("Profit", "sum")
    ).sort_values("profit", ascending=False).head(10)

    lines = [f"Top 10 products by total profit:"]
    for i, (name, row) in enumerate(top_profit.iterrows(), 1):
        lines.append(f"  {i}. {name}: profit {_fmt(row['profit'])}, sales {_fmt(row['sales'])}")
    docs.append({"text": "\n".join(lines), "metadata": {"type": "statistical", "topic": "top_products_profit"}})

    # Bottom 10 products by profit (loss-makers)
    bottom_profit = df.groupby("Product Name").agg(
        sales=("Sales", "sum"), profit=("Profit", "sum")
    ).sort_values("profit", ascending=True).head(10)

    lines = [f"Bottom 10 products by profit (biggest losses):"]
    for i, (name, row) in enumerate(bottom_profit.iterrows(), 1):
        lines.append(f"  {i}. {name}: profit {_fmt(row['profit'])}, sales {_fmt(row['sales'])}")
    docs.append({"text": "\n".join(lines), "metadata": {"type": "statistical", "topic": "bottom_products_profit"}})

    # Discount analysis
    disc_bins = pd.cut(df["Discount"], bins=[-0.001, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0],
                       labels=["No discount", "1-10%", "11-20%", "21-30%", "31-50%", "51-100%"],
                       include_lowest=True)
    disc_analysis = df.groupby(disc_bins, observed=True).agg(
        count=("Sales", "count"),
        avg_profit_margin=("Profit Margin", "mean"),
        total_sales=("Sales", "sum"),
    )
    lines = ["Discount impact analysis:"]
    for label, row in disc_analysis.iterrows():
        lines.append(
            f"  {label}: {int(row['count'])} transactions, "
            f"avg profit margin {row['avg_profit_margin']:.1f}%, "
            f"total sales {_fmt(row['total_sales'])}"
        )
    docs.append({"text": "\n".join(lines), "metadata": {"type": "statistical", "topic": "discount_analysis"}})

    # Segment analysis
    for seg, group in df.groupby("Segment"):
        sales = group["Sales"].sum()
        profit = group["Profit"].sum()
        margin = (profit / sales * 100) if sales > 0 else 0
        text = (
            f"Segment summary for {seg}: "
            f"Total sales {_fmt(sales)}, total profit {_fmt(profit)}, "
            f"profit margin {margin:.1f}%. "
            f"{group['Order ID'].nunique()} orders, {group['Customer ID'].nunique()} customers."
        )
        docs.append({"text": text, "metadata": {"type": "statistical", "topic": "segment", "segment": seg}})

    # Shipping mode analysis
    for mode, group in df.groupby("Ship Mode"):
        sales = group["Sales"].sum()
        profit = group["Profit"].sum()
        text = (
            f"Shipping mode summary for {mode}: "
            f"Total sales {_fmt(sales)}, total profit {_fmt(profit)}. "
            f"{len(group)} transactions ({len(group)/len(df)*100:.1f}% of all)."
        )
        docs.append({"text": text, "metadata": {"type": "statistical", "topic": "shipping", "ship_mode": mode}})

    # Year-over-year category trends
    cat_year = df.groupby(["Year", "Category"]).agg(
        sales=("Sales", "sum"), profit=("Profit", "sum")
    ).reset_index()
    for cat in df["Category"].unique():
        subset = cat_year[cat_year["Category"] == cat].sort_values("Year")
        lines = [f"Year-over-year trend for {cat}:"]
        for _, row in subset.iterrows():
            lines.append(f"  {int(row['Year'])}: sales {_fmt(row['sales'])}, profit {_fmt(row['profit'])}")
        docs.append({
            "text": "\n".join(lines),
            "metadata": {"type": "statistical", "topic": "category_trend", "category": cat},
        })

    # Monthly seasonality pattern
    monthly_avg = df.groupby("Month").agg(sales=("Sales", "sum")).reset_index()
    monthly_avg["sales"] = monthly_avg["sales"] / df["Year"].nunique()  # average per year
    best_month = monthly_avg.loc[monthly_avg["sales"].idxmax()]
    worst_month = monthly_avg.loc[monthly_avg["sales"].idxmin()]
    import calendar
    lines = ["Average monthly sales (seasonality pattern):"]
    for _, row in monthly_avg.iterrows():
        lines.append(f"  {calendar.month_name[int(row['Month'])]}: {_fmt(row['sales'])}")
    lines.append(f"Peak month: {calendar.month_name[int(best_month['Month'])]} ({_fmt(best_month['sales'])})")
    lines.append(f"Lowest month: {calendar.month_name[int(worst_month['Month'])]} ({_fmt(worst_month['sales'])})")
    docs.append({"text": "\n".join(lines), "metadata": {"type": "statistical", "topic": "seasonality"}})

    # Top 10 cities by sales
    top_cities = df.groupby(["City", "State", "Region"]).agg(
        sales=("Sales", "sum"), profit=("Profit", "sum")
    ).sort_values("sales", ascending=False).head(10).reset_index()
    lines = ["Top 10 cities by total sales:"]
    for i, row in top_cities.iterrows():
        lines.append(f"  {row['City']}, {row['State']} ({row['Region']}): sales {_fmt(row['sales'])}, profit {_fmt(row['profit'])}")
    docs.append({"text": "\n".join(lines), "metadata": {"type": "statistical", "topic": "top_cities"}})

    # Region comparison
    region_comp = df.groupby("Region").agg(
        sales=("Sales", "sum"), profit=("Profit", "sum"),
        orders=("Order ID", "nunique"), customers=("Customer ID", "nunique")
    ).sort_values("sales", ascending=False)
    lines = ["Region comparison (ranked by sales):"]
    for region, row in region_comp.iterrows():
        margin = (row["profit"] / row["sales"] * 100) if row["sales"] > 0 else 0
        lines.append(
            f"  {region}: sales {_fmt(row['sales'])}, profit {_fmt(row['profit'])}, "
            f"margin {margin:.1f}%, {int(row['orders'])} orders, {int(row['customers'])} customers"
        )
    docs.append({"text": "\n".join(lines), "metadata": {"type": "statistical", "topic": "region_comparison"}})

    print(f"Created {len(docs)} statistical summaries")
    return docs

# Main

def prepare_all(csv_path: str = "Sample - Superstore.csv", output_path: str = "prepared_chunks.json"):
    # Run full data preparation pipeline and save chunks to JSON.
    df = load_dataset(csv_path)

    all_docs = []
    all_docs.extend(create_transaction_descriptions(df))
    all_docs.extend(create_monthly_summaries(df))
    all_docs.extend(create_quarterly_summaries(df))
    all_docs.extend(create_yearly_summaries(df))
    all_docs.extend(create_category_summaries(df))
    all_docs.extend(create_regional_summaries(df))
    all_docs.extend(create_statistical_summaries(df))

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    # Print chunk size statistics
    sizes = [len(doc["text"]) for doc in all_docs]
    print(f"\n--- Chunk Statistics ---")
    print(f"Total documents: {len(all_docs)}")
    print(f"Avg chunk size: {np.mean(sizes):.0f} chars")
    print(f"Min chunk size: {np.min(sizes)} chars")
    print(f"Max chunk size: {np.max(sizes)} chars")
    print(f"Median chunk size: {np.median(sizes):.0f} chars")

    type_counts = {}
    for doc in all_docs:
        t = doc["metadata"]["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\nDocuments by type:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    print(f"\nSaved to {output_path}")
    return all_docs


if __name__ == "__main__":
    prepare_all()
