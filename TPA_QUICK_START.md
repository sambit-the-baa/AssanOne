# TPA Dashboard - Quick Start Guide

## 🚀 Get Started in 2 Minutes

### Step 1: Install Requirements
```bash
pip install streamlit
```

### Step 2: Run the Dashboard
```bash
streamlit run tpa_dashboard_with_login.py
```

### Step 3: Login
Your browser will open to `http://localhost:8501`

Use any of these demo accounts:
- **admin** / admin123
- **tpa_manager** / manager123
- **claims_reviewer** / reviewer123
- **auditor** / auditor123

---

## 📋 What You Can Do

### As TPA Manager or Admin:
1. ✅ View dashboard overview with KPIs
2. ✅ Upload and analyze claim documents
3. ✅ Process multiple claims at once
4. ✅ View fraud risk assessments
5. ✅ Generate reports and analytics
6. ✅ View provider performance
7. ✅ Detect fraud patterns

### As Admin Only:
- 👥 Manage user accounts
- 🔐 Create new users
- 📊 Assign roles and permissions

### As Auditor:
- 📈 View reports and analytics
- 🔍 Monitor fraud detection
- 📋 Access compliance data

---

## 🎯 Key Features

### Dashboard Overview
- Real-time KPI metrics
- Fraud detection statistics
- Claims processing status
- Recent activity feed

### Claims Management
**Analyze Single Claim:**
- Upload PDF or select from Data folder
- Choose analysis type
- Set priority level
- View detailed fraud assessment

**Bulk Processing:**
- Process multiple claims simultaneously
- Choose processing mode (Parallel, Sequential, Hybrid)
- Real-time progress tracking

**Claims History:**
- Filter by date, status, risk level
- Search historical records
- Export data

### Reports & Analytics
- **Fraud Reports** - Monthly trends, fraud types breakdown
- **Financial Summary** - Claims amounts, fraud losses, savings
- **Provider Analysis** - High-risk providers, legitimate providers
- **Pattern Detection** - Unusual claim patterns
- **Custom Reports** - Generate and export specialized reports

---

## 🔐 Security & Roles

### User Roles:
1. **Admin** - Full system access + user management
2. **TPA Manager** - Dashboard + claims + reports
3. **Claims Reviewer** - Dashboard + claims only
4. **Auditor** - Reports + analytics only

### Permissions:
- Dashboard access
- Claims management (upload, analyze, review)
- Report generation
- Analytics viewing
- User management (Admin only)

---

## 📊 Sample Dashboard Metrics

When you log in, you'll see:
- **2,847** Total Claims Processed
- **89** Fraud Cases Detected
- **$2.3M** Amount Protected
- **3.13%** Fraud Detection Rate

---

## 🔧 Common Tasks

### Upload a Claim
1. Go to **Claims Management** tab
2. Click **Analyze Single Claim**
3. Upload PDF or select from Data folder
4. Choose analysis type
5. Click **Analyze Claim**

### View Reports
1. Go to **Reports & Analytics**
2. Choose report type:
   - Fraud Reports
   - Financial Summary
   - Custom Reports
3. Click **Generate Report**
4. Download or email result

### Generate Custom Report
1. Go to **Reports > Custom Reports**
2. Select report type
3. Choose date range
4. Click **Generate Report**
5. Download as PDF or email

### Manage Users (Admin Only)
1. Go to **User Management**
2. View active users
3. Add new user with credentials and role
4. Click **Add User**

---

## 📁 Data Integration

The dashboard works with your existing fraud detection system:

**Data Location:** `c:\One Intelligcnc agents\Data\`

**Supported Files:**
- PDF claims documents
- JSON extracted data
- OCR results

**Processing Pipeline:**
- Upload → OCR → Fraud Detection → Risk Scoring → Report

---

## 🐛 Troubleshooting

### Dashboard Won't Start
```bash
# Check if port 8501 is available
streamlit run tpa_dashboard_with_login.py --server.port 8502
```

### Login Not Working
- Verify username/password exactly (case-sensitive)
- Check credentials in demo accounts above
- Clear browser cache

### Analysis Not Processing
- Ensure PDF file is valid
- Check file size (< 50MB recommended)
- Try smaller file first

### Need Help?
See [TPA_DASHBOARD_README.md](TPA_DASHBOARD_README.md) for detailed documentation

---

## 📱 Dashboard Pages

| Page | Access Level | Description |
|------|-------------|-------------|
| 📊 Dashboard | All | KPIs, metrics, activity |
| 📄 Claims | Manager+ | Upload, analyze, review |
| 📈 Reports | Manager+ | Fraud, financial, custom |
| 🔍 Analytics | Manager+ | Provider, patterns, intelligence |
| 👥 Users | Admin | User management, roles |

---

## 💡 Tips & Tricks

1. **Multi-tab Navigation** - Stay on same page while opening new tabs
2. **Keyboard Shortcuts** - Use Tab to navigate form fields
3. **Report Export** - Download reports as PDF or email directly
4. **Filters** - Use date range and status filters for quick searches
5. **Dark Mode** - Settings > Theme > Dark

---

## 🎓 Next Steps

1. ✅ Login with demo account
2. ✅ Explore dashboard metrics
3. ✅ Upload a sample claim
4. ✅ Run fraud analysis
5. ✅ View generated reports
6. ✅ Check user management

---

## 📞 Support

**Error? Not working?**
1. Check credentials
2. Ensure port 8501 is free
3. Try `streamlit cache clear`
4. Restart dashboard

**Feature Questions?**
- See full documentation in `TPA_DASHBOARD_README.md`
- Check integration guide for fraud detection system

---

**Version:** 1.0.0  
**Last Updated:** December 30, 2025  
**Platform:** Windows 10/11, Linux, macOS
