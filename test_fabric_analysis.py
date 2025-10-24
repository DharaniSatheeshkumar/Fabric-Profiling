from Fabric_profiling import GeminiFabricRecognizer
import os

def test_recognizer(api_key):
    # Create test directory for sample images
    os.makedirs("sample_images", exist_ok=True)
    
    try:
        # Initialize the recognizer with provided API key
        recognizer = GeminiFabricRecognizer(api_key=api_key)
        
        print("\n✅ Successfully initialized GeminiFabricRecognizer")
        print(f"🔑 API Key is valid")
        print(f"📁 Sample images directory: {os.path.abspath('sample_images')}")
        print("\nTo analyze fabric images:")
        print("1. Add fabric images to the 'sample_images' directory")
        print("2. Run the main script with:")
        print("   python Fabric_profiling.py --image sample_images/your_image.jpg")
        return True
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    api_key = input("Enter your Google API key: ").strip()
    test_recognizer(api_key)