🚀 Smart Salary Prediction System
📌 Overview

This project presents an AI-powered system that provides personalized job recommendations along with expected salary predictions. It leverages multiple datasets such as job skills, industries, company data, and salaries to deliver accurate, data-driven insights for job seekers.

🎯 Problem Statement

Finding the right job that matches a candidate’s skills, experience, and expectations is difficult. Traditional platforms provide generic recommendations without personalization, leading to inefficient job searches and mismatched applications.

💡 Solution

This system uses Machine Learning to:

Analyze user skills, experience, and education
Match them with job requirements
Recommend the top 3 relevant jobs
Predict expected salary
🧠 Key Features
✅ Personalized job recommendations
💰 Salary prediction
📊 Skill-based matching using TF-IDF
🏢 Industry-based filtering
🎯 Top 3 job suggestions
⚡ Fast and efficient predictions
🏗️ System Workflow
User Input → Data Preprocessing → Feature Engineering → ML Model → Output
🛠️ Technology Stack
Python
Pandas, NumPy
Scikit-learn
XGBoost
TF-IDF Vectorizer
Streamlit (optional UI)
📂 Datasets Used
job_skills.csv
skills.csv
job_industries.csv
industries.csv
company_industries.csv
employee_counts.csv
salaries.csv
postings.csv
⚙️ Implementation Details
Data cleaned and merged from multiple sources
Feature engineering:
Skill count, ratios, experience
Education extracted from job titles
TF-IDF used for text vectorization
Models used:
Random Forest
XGBoost
Best model selected using R² score
Cosine similarity used for job recommendation
📊 Output
💰 Predicted Salary
🎯 Top 3 Job Recommendations
▶️ How to Run
Place all CSV files in the same folder

Install dependencies:

pip install pandas numpy scikit-learn xgboost

Run the script:

python model.py
Enter:
Skills
Experience
Education
📈 Example

Input:

Skills: python sql machine learning
Experience: 3
Education: 2

Output:

Expected Salary: ₹8,50,000
Top Jobs:
- Data Analyst
- ML Engineer
- Software Engineer
🌍 Impact
Improves job search efficiency
Provides realistic salary expectations
Helps users make informed career decisions
🔮 Future Improvements
Resume parsing
Deep learning (BERT embeddings)
Real-time job APIs
Advanced recommendation system
Web deployment
🏁 Conclusion

This project demonstrates how machine learning can enhance job search platforms by providing personalized and intelligent recommendations, improving both efficiency and user experience.

👨‍💻 Team
Team Name: Team Invictus
Institution: Presidency University
