import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "../data/processed_pavement_data.csv"


# Load processed dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# Check required columns
required_columns = ["seg_id", "pci", "pci_desc", "maintenance_priority"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")


# -----------------------------
# 1. PCI Distribution
# -----------------------------

plt.figure(figsize=(10, 6))

plt.hist(df["pci"], bins=20)

plt.xlabel("Pavement Condition Index (PCI)")
plt.ylabel("Number of Road Segments")
plt.title("Distribution of Pavement Condition Index")

plt.tight_layout()
plt.savefig("../pci_distribution.png", dpi=300)
plt.show()


# -----------------------------
# 2. Pavement Condition Distribution
# -----------------------------

condition_counts = df["pci_desc"].value_counts()

plt.figure(figsize=(10, 6))

condition_counts.plot(kind="bar")

plt.xlabel("Pavement Condition")
plt.ylabel("Number of Road Segments")
plt.title("Pavement Condition Distribution")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("../condition_distribution.png", dpi=300)
plt.show()


# -----------------------------
# 3. Maintenance Priority
# -----------------------------

priority_counts = df["maintenance_priority"].value_counts()

plt.figure(figsize=(10, 6))

priority_counts.plot(kind="bar")

plt.xlabel("Maintenance Priority")
plt.ylabel("Number of Road Segments")
plt.title("Maintenance Priority Distribution")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("../maintenance_priority.png", dpi=300)
plt.show()


# -----------------------------
# 4. PCI by Maintenance Priority
# -----------------------------

plt.figure(figsize=(10, 6))

df.boxplot(
    column="pci",
    by="maintenance_priority"
)

plt.xlabel("Maintenance Priority")
plt.ylabel("PCI")
plt.title("PCI by Maintenance Priority")

plt.suptitle("")

plt.tight_layout()
plt.savefig("../pci_by_priority.png", dpi=300)
plt.show()


print("\nVisualization completed successfully.")
print("Charts saved in the project root folder.")
