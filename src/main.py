from src.model import train_and_save_model


def main():
    model_path = train_and_save_model("data.pkl")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
