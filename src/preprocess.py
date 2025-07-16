import pandas as pd

def generate_supplier_overview(input_csv_path: str, output_csv_path: str) -> None:
    df = pd.read_csv(input_csv_path)

    # Convert date columns to datetime
    df['expected_delivery_date'] = pd.to_datetime(df['Expected Delivery Date'])
    df['actual_delivery_date'] = pd.to_datetime(df['Actual Delivery Date'])

    # Calculate delay in days
    df['delay_days'] = (df['actual_delivery_date'] - df['expected_delivery_date']).dt.days
    df['is_delayed'] = df['delay_days'] > 0

    # Group by supplier and value category
    summary = (
        df.groupby(['Supplier ID', 'Value Category'])
          .agg(
              total_deliveries=('is_delayed', 'count'),
              delayed_deliveries=('is_delayed', 'sum'),
              avg_delay=('delay_days', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
              avg_shipment_volume=('Shipment Volume', 'mean'),
              pct_shipment_lost=('Shipment Lost', 'mean'),
              pct_defect_rate=('Defected', 'mean')
          )
          .reset_index()
    )

    # Calculate percentages
    summary['%_on_time'] = 100 * (summary['total_deliveries'] - summary['delayed_deliveries']) / summary['total_deliveries']
    summary['%_delayed'] = 100 * summary['delayed_deliveries'] / summary['total_deliveries']

    # Select final columns
    final_df = summary[['Supplier ID', 'Value Category', '%_on_time', '%_delayed', 'avg_delay', 
                        'avg_shipment_volume', 'pct_shipment_lost', 'pct_defect_rate']]

    # Save to CSV
    final_df.to_csv(output_csv_path, index=False)
    print(f"supplier_overview saved to {output_csv_path}")