# TPA Dashboard with Login Portal

A comprehensive Third Party Administrator (TPA) management dashboard with secure login, claims processing, fraud detection analytics, and user management.

## Features

### 🔐 Authentication & Security
- **Secure Login Portal** - Username/password authentication
- **Role-Based Access Control** - Different permissions for different user types
- **User Management** - Admin panel for managing users and roles
- **Activity Logging** - Track user actions and login history

### 📊 Dashboard Overview
- **KPI Metrics** - Real-time claims and fraud statistics
- **Claims Status** - Visual breakdown of claim states
- **Fraud Risk Distribution** - Risk level analysis
- **Recent Activity Feed** - Latest system events

### 📄 Claims Management
- **Single Claim Analysis**
  - Upload or select from local files
  - Multiple analysis types (Full, Quick, OCR, Pattern Detection)
  - Priority level assignment
  - Detailed risk assessment with factors
  
- **Bulk Processing**
  - Process multiple claims simultaneously
  - Choose processing mode (Parallel, Sequential, Hybrid)
  - Real-time progress tracking
  
- **Claims History**
  - Filter by date, status, and risk level
  - Search and review historical claims
  - Export capabilities

### 📈 Reports & Analytics
- **Fraud Detection Reports**
  - Monthly fraud trends
  - Top fraud categories breakdown
  - Case statistics
  
- **Financial Summary**
  - Total claims processed
  - Fraud loss amounts
  - Monthly financial trends
  
- **Custom Reports**
  - Generate specialized reports
  - Export to PDF
  - Email distribution
  
- **Provider Analysis**
  - High-risk provider identification
  - Top legitimate providers
  - Performance metrics
  
- **Claim Pattern Detection**
  - Unusual pattern identification
  - Pattern frequency analysis
  - Risk assessment
  
- **Risk Intelligence**
  - Predictive alerts
  - Active investigations
  - License renewal tracking

### 👥 User Management (Admin Only)
- Manage user accounts
- Assign roles and permissions
- Track user activity
- Monitor login history

## User Roles & Permissions

### Admin
- Access to all features
- User management
- Full reporting
- System configuration

### TPA Manager
- Dashboard access
- Claims management
- View all reports
- Fraud analytics

### Claims Reviewer
- Dashboard access
- Claims analysis and review
- Claim history

### Auditor
- View reports
- Access analytics
- Compliance monitoring
- Cannot modify claims

## Demo Credentials

| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Administrator |
| tpa_manager | manager123 | TPA Manager |
| claims_reviewer | reviewer123 | Claims Reviewer |
| auditor | auditor123 | Auditor |

## Installation

### Prerequisites
- Python 3.8+
- Streamlit
- pip package manager

### Setup Steps

1. **Install Dependencies**
```bash
pip install streamlit
# Or install all requirements
pip install -r requirements-google-agents.txt
```

2. **Run the Dashboard**
```bash
streamlit run tpa_dashboard_with_login.py
```

3. **Access the Application**
- Open your browser to `http://localhost:8501`
- Login with demo credentials (see above)

## File Structure

```
c:\One Intelligcnc agents\
├── tpa_dashboard_with_login.py    # Main dashboard with login
├── fraud_detection_dashboard.py   # Original fraud detection dashboard
├── streamlit_app.py               # OCR extraction dashboard
├── requirements-google-agents.txt # Dependencies
├── Data/                          # Sample data
│   └── outputs/                   # Processed claim outputs
└── agent/                         # Fraud detection agents
    ├── agents_fraud.py
    ├── pipeline.py
    └── ...
```

## Running Multiple Dashboards

If you want to run both dashboards:

### Terminal 1 - TPA Dashboard (Recommended)
```bash
streamlit run tpa_dashboard_with_login.py
```

### Terminal 2 - Original Fraud Detection Dashboard
```bash
streamlit run fraud_detection_dashboard.py --logger.level=error
```

### Terminal 3 - OCR Dashboard
```bash
streamlit run streamlit_app.py --logger.level=error
```

Each will run on different ports:
- TPA Dashboard: `http://localhost:8501`
- Fraud Detection: `http://localhost:8502`
- OCR Dashboard: `http://localhost:8503`

## Dashboard Pages

### 📊 Dashboard
Main overview with key performance indicators and recent activity

### 📄 Claims Management
- Analyze individual claims
- Process claims in bulk
- View claims history and filters

### 📈 Reports & Analytics
- Fraud detection reports with trends
- Financial summaries
- Custom report generation
- Provider performance analysis
- Claim pattern detection
- Risk intelligence

### 👥 User Management (Admin Only)
- Manage user accounts
- Assign roles and permissions
- View activity logs

## Customization

### Adding New Users
1. Go to User Management (Admin only)
2. Click "Add New User"
3. Enter username, password, and role
4. Click "Add User"

### Modifying Credentials
Edit the `CREDENTIALS` dictionary in the `SimpleAuth` class:
```python
CREDENTIALS = {
    "username": "password",
    "another_user": "another_password"
}
```

### Updating Roles & Permissions
Modify the `ROLES` dictionary:
```python
ROLES = {
    "role_name": ["permission1", "permission2", ...]
}
```

## Security Notes

**Important**: This is a demo implementation with hardcoded credentials. For production:

1. **Use a Real Database**
   - PostgreSQL, MongoDB, or similar
   - Store hashed passwords using bcrypt/argon2
   
2. **Implement SSL/TLS**
   - Use HTTPS for all communications
   
3. **Add Session Management**
   - Implement session timeouts
   - Add CSRF protection
   
4. **Use OAuth/SAML**
   - Integrate with enterprise identity providers
   - Multi-factor authentication
   
5. **Audit Logging**
   - Store all user actions in secure logs
   - Implement compliance monitoring

## Sample Data Integration

The dashboard can integrate with your existing fraud detection system:

```python
# In claims analysis tab, integrate with:
from agent.pipeline import process_claim_full_pipeline

result = process_claim_full_pipeline(
    pdf_path,
    execution_mode="parallel",
    save_results=True
)
```

## Troubleshooting

### Port Already in Use
```bash
streamlit run tpa_dashboard_with_login.py --server.port 8504
```

### Session Issues
Clear Streamlit cache:
```bash
streamlit cache clear
```

### Login Not Working
- Check username/password (case-sensitive)
- Verify credentials in `SimpleAuth.CREDENTIALS`
- Check browser console for errors

## Future Enhancements

- [ ] Database integration for user storage
- [ ] LDAP/Active Directory integration
- [ ] Two-factor authentication
- [ ] Real-time claim processing with WebSockets
- [ ] Advanced data visualization
- [ ] Email notifications
- [ ] API integration with legacy systems
- [ ] Compliance reporting (HIPAA, SOX)
- [ ] Machine learning fraud prediction
- [ ] Provider network analysis

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Streamlit documentation: https://docs.streamlit.io
3. Check fraud detection agents documentation in FRAUD_DETECTION_SYSTEM.md

## License

Your project license here

---

**Last Updated**: December 30, 2025
**Dashboard Version**: 1.0.0
