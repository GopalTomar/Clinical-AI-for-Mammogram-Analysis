"""
Test script to verify Formspree integration
Run this separately to debug email sending issues
"""

import requests
import json

# Your Formspree endpoint
FORMSPREE_ENDPOINT = "https://formspree.io/f/xpqjdwqv"  # CV Project form

def test_formspree():
    """Test Formspree connection with detailed debugging"""
    
    print("=" * 60)
    print("FORMSPREE CONNECTION TEST")
    print("=" * 60)
    
    # Test data
    test_data = {
        "name": "Test User",
        "email": "test@example.com",
        "message": "This is a test message from the debugging script.",
        "_subject": "Test Submission"
    }
    
    print("\n1. Testing endpoint:", FORMSPREE_ENDPOINT)
    print("\n2. Sending test data:")
    print(json.dumps(test_data, indent=2))
    
    try:
        print("\n3. Making POST request...")
        response = requests.post(
            FORMSPREE_ENDPOINT,
            json=test_data,  # Use json= not data=
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0"
            },
            timeout=10
        )
        
        print(f"\n4. Response Status Code: {response.status_code}")
        print(f"5. Response Headers: {dict(response.headers)}")
        
        print("\n6. Response Body:")
        try:
            response_json = response.json()
            print(json.dumps(response_json, indent=2))
        except:
            print(response.text)
        
        print("\n" + "=" * 60)
        if response.status_code == 200:
            print("✅ SUCCESS! Email should be sent.")
            print("Check your Formspree dashboard:")
            print("https://formspree.io/forms/mjgeewjp/submissions")
        elif response.status_code == 422:
            print("❌ VALIDATION ERROR")
            print("Formspree rejected the submission.")
            print("Common causes:")
            print("  - Email address format invalid")
            print("  - Form not verified/activated")
            print("  - Missing required fields")
        elif response.status_code == 429:
            print("❌ RATE LIMIT EXCEEDED")
            print("You've exceeded Formspree's rate limits.")
            print("Wait a few minutes and try again.")
        elif response.status_code == 404:
            print("❌ FORM NOT FOUND")
            print("The Formspree endpoint doesn't exist.")
            print("Check if the form ID 'mjgeewjp' is correct.")
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print("Check the response body above for details.")
        print("=" * 60)
        
    except requests.exceptions.Timeout:
        print("\n❌ TIMEOUT ERROR")
        print("The request took too long (>10 seconds).")
        print("Possible causes:")
        print("  - Slow internet connection")
        print("  - Formspree server issues")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ CONNECTION ERROR")
        print("Could not connect to Formspree.")
        print("Possible causes:")
        print("  - No internet connection")
        print("  - Firewall blocking the request")
        print("  - Formspree is down")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")

def test_with_anonymous_email():
    """Test with the anonymous email used in feedback"""
    
    print("\n" + "=" * 60)
    print("TESTING WITH ANONYMOUS EMAIL")
    print("=" * 60)
    
    test_data = {
        "name": "Anonymous Feedback",
        "email": "anonymous@feedback.formspree.io",
        "message": "Testing with anonymous feedback email",
        "_subject": "Anonymous Feedback Test"
    }
    
    print("\nUsing email: anonymous@feedback.formspree.io")
    
    try:
        response = requests.post(
            FORMSPREE_ENDPOINT,
            json=test_data,  # Use json= not data=
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0"
            },
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Anonymous email works!")
        else:
            print(f"❌ Anonymous email failed: {response.status_code}")
            try:
                print(response.json())
            except:
                print(response.text)
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def check_internet_connection():
    """Check if you have internet connectivity"""
    
    print("\n" + "=" * 60)
    print("CHECKING INTERNET CONNECTION")
    print("=" * 60)
    
    try:
        response = requests.get("https://www.google.com", timeout=5)
        print("✅ Internet connection is working")
        return True
    except:
        print("❌ No internet connection detected")
        return False

if __name__ == "__main__":
    print("\n🔧 FORMSPREE DEBUGGING TOOL\n")
    
    # Check internet first
    if not check_internet_connection():
        print("\n⚠ Please check your internet connection and try again.")
        exit(1)
    
    # Run tests
    test_formspree()
    test_with_anonymous_email()
    
    print("\n\n📋 TROUBLESHOOTING CHECKLIST:")
    print("=" * 60)
    print("[ ] Is your internet connection working?")
    print("[ ] Is the Formspree form ID correct (mjgeewjp)?")
    print("[ ] Have you verified your Formspree email?")
    print("[ ] Is your Formspree account active?")
    print("[ ] Have you exceeded the free tier limit (50/month)?")
    print("[ ] Check Formspree status: https://status.formspree.io")
    print("=" * 60)
    
    print("\n\n💡 NEXT STEPS:")
    print("1. Check Formspree dashboard for submissions")
    print("2. Verify your email in Formspree settings")
    print("3. Check spam folder for Formspree verification emails")
    print("4. Review Formspree documentation: https://help.formspree.io")
    print("\n")