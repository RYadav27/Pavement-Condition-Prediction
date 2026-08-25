import pandas as pd


def load_and_clean_data(file_path):
    # Load CSV file
    df = pd.read_csv(file_path)

    print("Original dataset shape:", df.shape)

    # Keep required columns
    df = df[["seg_id", "pci", "pci_desc"]]

    # Convert PCI to numeric
    df["pci"] = pd.to_numeric(df["pci"], errors="coerce")

    # Remove rows with missing PCI
    df = df.dropna(subset=["pci"])

    # Remove invalid PCI values
    # PCI should be between 0 and 100
    df = df[(df["pci"] >= 0) & (df["pci"] <= 100)]

    # Remove duplicate records
    df = df.drop_duplicates()

    # Reset index
    df = df.reset_index(drop=True)

    # Create maintenance priority
    def assign_priority(pci):
        if pci < 25:
            return "Critical"
        elif pci < 40:
            return "High"
        elif pci < 55:
            return "Medium"
        elif pci < 70:
            return "Low"
        else:
            return "Routine"

    df["maintenance_priority"] = df["pci"].apply(assign_priority)

    print("Cleaned dataset shape:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nPavement condition distribution:")
    print(df["pci_desc"].value_counts())

    print("\nMaintenance priority distribution:")
    print(df["maintenance_priority"].value_counts())

    return df


if __name__ == "__main__":

    # Path to input dataset
    file_path = "../data/pavement_data.csv"

    # Process data
    cleaned_data = load_and_clean_data(file_path)

    # Save processed dataset
    output_path = "../data/processed_pavement_data.csv"

    cleaned_data.to_csv(output_path, index=False)

    print("\nProcessed data saved to:")
    print(output_path)
