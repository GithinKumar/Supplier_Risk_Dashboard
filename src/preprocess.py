import pandas as pd

def generate_supplier_overview(input_csv_path: str, output_csv_path: str) -> None:
    df = pd.read_csv(input_csv_path)

    # Convert date columns to datetime
    df['expected_delivery_date'] = pd.to_datetime(df['Expected Delivery Date'])
    df['actual_delivery_date'] = pd.to_datetime(df['Actual Delivery Date'])

    # Calculate delay in days
    df['delay_days'] = (df['actual_delivery_date'] - df['expected_delivery_date']).dt.days
    df['is_delayed'] = df['delay_days'] > 0

    # Add a Quarter column based on expected delivery date (you could use actual if you prefer)
    df['Quarter'] = df['expected_delivery_date'].dt.to_period('Q').dt.end_time
    df['Quarter'] = df['Quarter'].dt.strftime('%Y-%m-%d')

    # Group by supplier, quarter, and value category
    summary = (
        df.groupby(['Supplier ID', 'Quarter', 'Value Category'])
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
    summary['%_shipment_lost'] = summary['pct_shipment_lost'] * 100
    summary['%_defect_rate'] = summary['pct_defect_rate'] * 100

    # Select final columns
    final_df = summary[['Supplier ID', 'Quarter', 'Value Category', '%_on_time', '%_delayed', 'avg_delay', 
                        'avg_shipment_volume', '%_shipment_lost', '%_defect_rate']]

    # Save to CSV
    final_df.to_csv(output_csv_path, index=False)
    print(f"supplier_overview saved to {output_csv_path}")