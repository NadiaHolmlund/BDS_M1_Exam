# Imports
import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.sparse as ss
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.express as px
import plotly as pt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder 



# Before loading the data, we specify page configurations.
# Page_title and page_icon determines the layout of the webbrowser-tab.
# Layout determines the overall layout of the page. Wide specifies that the whole screen width is used for displaying the app.
st.set_page_config(
    page_title="JobHunter 🔍",
    page_icon="🔍",
    layout="wide")


# Using experimental_singleton and a definition, the datasets and static calculations are loaded only once in order to minimise processing time.
# The process is repeated for each dataset. Each definition includes data cleaning and preprocessing for the given dataset,
# which is copy/pasted from the notebook.

# Loading jobs.csv
@st.experimental_singleton
def load_data_jobs():
    # Import the csv file from Github
    jobs = pd.read_csv('https://raw.githubusercontent.com/NadiaHolmlund/BDS_M1_Exam/main/Data/jobs.csv')

    # Cleaning and preprocessing copy/pasted from the notebook
    jobs = jobs.drop(['Industry', 'Salary', 'Address', 'Requirements'], axis=1)
    jobs = jobs.dropna()
    jobs['Listing.Start'] = pd.to_datetime(jobs['Listing.Start'], format='%d/%m/%Y')
    jobs['Listing.End'] = pd.to_datetime(jobs['Listing.End'])
    jobs['Created.At'] = pd.to_datetime(jobs['Created.At'])
    jobs['Updated.At'] = pd.to_datetime(jobs['Updated.At'])
    jobs['Date'] = pd.to_datetime(jobs['Created.At']).dt.date
    jobs['Employment.Type'] = jobs['Employment.Type'].replace(['Seasonal/Temp'], 'Temporary/seasonal')
    jobs['Employment.Type'] = jobs['Employment.Type'].fillna('Unspecified')
    jobs['Education.Required'] = jobs['Education.Required'].fillna('Unspecified')
    jobs['Education.Required'] = jobs['Education.Required'].replace(['Not Specified'] , 'Unspecified')

    # Grouping positions into larger categories
    # Customer service
    jobs['Position'] = jobs['Position'].replace(['Customer Service Representative'] , 'Customer Service')
    jobs['Position'] = jobs['Position'].replace(['Customer Service / Sales ( New Grads Welcome!)'] , 'Customer Service')
    jobs['Position'] = jobs['Position'].replace(['Customer Service / Sales ( New Grads Welcome! )'] , 'Customer Service')
    jobs['Position'] = jobs['Position'].replace(['Entry Level Sales / Customer Service – Part time / Full Time'] , 'Customer Service')

    # Accounting
    jobs['Position'] = jobs['Position'].replace(['Accounts Payable Clerk'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Accounting Clerk'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Accounts Receivable Clerk'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Bookkeeper'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Full Charge Bookkeeper'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Payroll Clerk'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Billing Clerk'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Payroll Administrator'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Staff Accountant'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Senior Accountant'] , 'Accounting')

    # Sales
    jobs['Position'] = jobs['Position'].replace(['Sales Representative / Sales Associate ( Entry Level )'] , 'Sales')
    jobs['Position'] = jobs['Position'].replace(['Sales Associate'] , 'Sales')
    jobs['Position'] = jobs['Position'].replace(['Seasonal Wedding Sales Stylist'] , 'Sales')
    jobs['Position'] = jobs['Position'].replace(['Retail Sales Associate - Part-Time'] , 'Sales')
    jobs['Position'] = jobs['Position'].replace(['Retail Sales Associate'] , 'Sales')

    # Administration
    jobs['Position'] = jobs['Position'].replace(['Administrative Assistant'] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['Receptionist'] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['Front Desk Coordinator'] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['Executive Assistant'] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['General Office Clerk '] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['Office Assistant'] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['Medical Receptionist'] , 'Administration')

    # Restaurant
    jobs['Position'] = jobs['Position'].replace(['Bartender'] , 'Restaurant personnel')
    jobs['Position'] = jobs['Position'].replace(['Server'] , 'Restaurant personnel')
    jobs['Position'] = jobs['Position'].replace(['Kitchen Staff'] , 'Restaurant personnel')
    jobs['Position'] = jobs['Position'].replace(['Hiring All Restaurant Positions - Servers - Cooks - Bartenders'] , 'Restaurant personnel')
    jobs['Position'] = jobs['Position'].replace(['Cook'] , 'Restaurant personnel')
    jobs['Position'] = jobs['Position'].replace(['Kitchen Staff'] , 'Restaurant personnel')

    # Caregiving
    jobs['Position'] = jobs['Position'].replace(['Caregiver / Home Health Aide / CNA'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Registered Nurse'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Home Health Aide'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Certified Nursing Assistant'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Entry Level Caregiver / Home Health Aide'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Caregiving'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Entry Level Caregiver'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Entry Level Healthcare Professionals wanted for Caregiver Opportunities'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Registered Nurse (RN) / Licensed Practical Nurse (LPN) - Healthcare Nursing Staff'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Licensed Practical Nurse - LPN'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Physical Therapist'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Certified Nursing Assistant (CNA) - Healthcare Nursing Staff'] , 'Caregiving professional')

    # Human resources
    jobs['Position'] = jobs['Position'].replace(['Human Resources Assistant'] , 'Human resources')
    jobs['Position'] = jobs['Position'].replace(['Human Resources Recruiter'] , 'Human resources')

    # Retail professional
    jobs['Position'] = jobs['Position'].replace(['Retail Sales Associate – Part-Time'] , 'Retail professional')
    jobs['Position'] = jobs['Position'].replace(['Retail Sales Associate / Photographer'] , 'Retail professional')
    jobs['Position'] = jobs['Position'].replace(['Retail Sales Associate - Part Time'] , 'Retail professional')
    jobs['Position'] = jobs['Position'].replace(['Part Time Retail Merchandiser'] , 'Retail professional')
    jobs['Position'] = jobs['Position'].replace(['Sales Representative - Retail'] , 'Retail professional')

    # Teacher 
    jobs['Position'] = jobs['Position'].replace(['Assistant Teacher'] , 'Teacher employee')
    jobs['Position'] = jobs['Position'].replace(['Teacher'] , 'Teacher employee')

    # Security officer
    jobs['Position'] = jobs['Position'].replace(['Security Officer'] , 'Security officer')
    jobs['Position'] = jobs['Position'].replace(['Security Officer - Regular'] , 'Security officer')

    # Driver professional
    jobs['Position'] = jobs['Position'].replace(['Part Time School Bus Drivers WANTED - Training Available'] , 'Driver professional')
    jobs['Position'] = jobs['Position'].replace(['Delivery Driver (Part -Time)'] , 'Driver professional')
    jobs['Position'] = jobs['Position'].replace(['School Bus Driver'] , 'Driver professional')
    jobs['Position'] = jobs['Position'].replace(['Driver'] , 'Driver professional')

    # Reducing the dataset to includes only rows where position is included in the categories specified above
    jobs = jobs[jobs['Position'].isin(['Customer Service', 'Accounting','Sales', 'Administration', 'Restaurant personnel', 'Caregiving professional', 'Human resources', 'Retail professional', 'Teacher employee','Security officer', 'Driver professional'])]
    
    # calculating various metrics
    open_positions = jobs['Job.ID'].nunique()
    companies_hiring = jobs['Company'].nunique()

    # calculating the top 5 companies and positions posted
    top_5_companies = jobs['Company'].value_counts().nlargest(5)
    top_5_positions = jobs['Position'].value_counts().nlargest(5)

    # Interactive timelines
    jobs_pivot = pd.pivot_table(jobs, values='Job.ID', index='Date', columns='Employment.Type', aggfunc='count')
    positions_pivot = pd.pivot_table(jobs, values='Job.ID', index='Date', columns='Position', aggfunc='count')

    return jobs, open_positions, companies_hiring, top_5_companies, top_5_positions, jobs_pivot, positions_pivot


# Loading user_job_views.csv
@st.experimental_singleton
def load_data_user_view():
    # Import the csv file from Github
    user_view = pd.read_csv('https://raw.githubusercontent.com/NadiaHolmlund/BDS_M1_Exam/main/Data/user_job_views.csv')

    # Cleaning and preprocessing copy/pasted from the notebook
    user_view = user_view.drop(['Industry'], axis=1)
    user_view['Company'] = user_view['Company'].fillna('Unspecified')
    user_view  = user_view.dropna(subset=['State.Name'])
    user_view['View.Duration'] = user_view['View.Duration'].fillna(user_view['View.Duration'].mean())
    user_view['Created.At'] = pd.to_datetime(user_view['Created.At'])
    user_view['Updated.At'] = pd.to_datetime(user_view['Updated.At'])
    user_view['View.Start'] = pd.to_datetime(user_view['View.Start'])
    user_view['View.End'] = pd.to_datetime(user_view['View.End'])
    user_view['View.Duration'] = user_view['View.Duration'].mask(((user_view['View.Duration']< user_view['View.Duration'].quantile(0.05)) | (user_view['View.Duration'] > user_view['View.Duration'].quantile(0.95))), np.nan)
    imputer = SimpleImputer(missing_values=np.NaN, strategy='median')
    user_view['View.Duration'] = imputer.fit_transform(user_view['View.Duration'].values.reshape(-1,1))[:,0]

    # Processing data for the UML recommender
    # Encode ids
    le_applicant = LabelEncoder()
    le_title = LabelEncoder()
    user_view['applicant_id'] = le_applicant.fit_transform(user_view['Applicant.ID'])
    user_view['title_id'] = le_title.fit_transform(user_view['Title'])

    # Construct matrix
    ones = np.ones(len(user_view), np.uint32)
    matrix = ss.coo_matrix((ones, (user_view['applicant_id'], user_view['title_id'])))

    # Decomposition
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    matrix_applicant = svd.fit_transform(matrix)
    matrix_title = svd.fit_transform(matrix.T)

    # Distance-matrix
    cosine_distance_matrix_title = cosine_distances(matrix_title)

    # calculating various metrics
    active_jobhunters = user_view['Applicant.ID'].nunique()
    avg_job_posts_viewed_per_jobhunter = (user_view['Job.ID'].value_counts().sum() / user_view['Applicant.ID'].nunique()).round(2)
    avg_viewtime_per_job_post = (user_view['View.Duration'].mean() / 60).round(2)

    return user_view, le_applicant, le_title, matrix, svd, matrix_applicant, matrix_title, cosine_distance_matrix_title, active_jobhunters, avg_job_posts_viewed_per_jobhunter, avg_viewtime_per_job_post, imputer

# Loading our SML data into streamlit
# Using experimentatl_singleton and a definition, we load the data only once in order to minimise processing time
@st.experimental_singleton
def load_data_sml():
    data_sml_x = pd.read_csv('https://raw.githubusercontent.com/NadiaHolmlund/BDS_M1_Exam/main/SML_Data/data_sml_x.csv')  
    y = pd.read_csv('https://raw.githubusercontent.com/NadiaHolmlund/BDS_M1_Exam/main/SML_Data/data_sml_y.csv')
    rf_pickle = pickle.load(open("SML_Data/random_forest_rec.pickle", "rb")) 
    ordinal_pickle = pickle.load(open("SML_Data/ordinal_enc.pickle", "rb"))
    label_pickle = pickle.load(open("SML_Data/label_enc.pickle", "rb"))

  
    return data_sml_x, y, rf_pickle, ordinal_pickle, label_pickle


# Loading user__past_experience_views.csv
# @st.experimental_singleton
# def load_data_user_exp():
    # Import the csv file from Github
    # user_exp = pd.read_csv('https://raw.githubusercontent.com/NadiaHolmlund/BDS_M1_Exam/main/Data/user_past_experience.csv')

    # return user_exp


# Loading user_work_interest.csv
# @st.experimental_singleton
# def load_data_user_int():
    # Import the csv file from Github
    # user_int = pd.read_csv('https://raw.githubusercontent.com/NadiaHolmlund/BDS_M1_Exam/main/Data/user_work_interest.csv')

    # return user_int


# Loading recommender_selectbox
@st.experimental_singleton
def load_recommender_selectbox():
    recommender_selectbox = pd.DataFrame(["Server @ Haven", "Server @ Oola Restaurant & Bar", "Server @ Burma Superstar", 
                                      "Server @ The Liberties Bar & Restaurant", "Server @ Sanraku Metreon", "Server @ COCO5OO", 
                                      "Server @ A La Turca", "Server @ The Liberty Cafe", "Server @ Yemeni's Restaurant", "Server @ L'Olivier", 
                                      "Waitstaff / Server @ Atria Senior Living", "Part Time Showroom Sales / Cashier @ Grizzly Industrial Inc.", 
                                      "Receptionist @ confidential", "Coordinator/Scheduler - IT @ Integrated Systems Analysts, Inc.", 
                                      "COMMUNITY ASSISTANT", "Part Time Errand/Clerical Assistant", "PART-TIME Administrative Assistant", 
                                      "Package Handler - Part-Time @ UPS", "Temporary Drivers @ Kelly Services", 
                                      "Customer Service Representative-Moonlighter @ U-Haul", "Pick-up Associate @ Orchard Supply Hardware", 
                                      "Part Time Liaison/Courier @ CIBTvisas.com", "NABISCO Part Time Merchandiser- Tucson 311 @ Mondelez International-Sales", 
                                      "Full Charge Bookkeeper Needed! @ Accountemps", "Entry Level Financial Analyst-Strong Excel Needed-Project! @ Accountemps",
                                      "Accountant @ Accountemps", "Accounting Manager / Supervisor @ Accountemps", "Accounts Payable Supervisor/Manager @ Accountemps",
                                      "Part Time Bookkeeper @ Accountemps", "Part Time Administrative Position in Omaha! @ Kelly Services", 
                                      "Mail Room Clerk @ OfficeTeam", "General Office Clerk @ OfficeTeam Healthcare", "DELIVERY DRIVERS @ Round Table Pizza",
                                      "Part-time School Bus Driver @ FirstGroup America", "92G Food Service Specialist @ Army National Guard", "School Bus Driver @ First Student", 
                                      "Business Consultants / Account Executives / Sales / (Inc.500/5000 Company) @ Central Payment", "Database Developer @ Spherion Staffing Services", 
                                      "Jr. Administrative Assistant @ OfficeTeam", 
                                      "Staff Nurse III @ University Health System", "Respiratory Therapist I @ University Health System", 
                                      "Administrative Assistant", "Part Time / Administrative/General Office - Part Time Administrative Assistant @ JobGiraffe",
                                      "Administrative Assistant - PT @ FCX Performance", "Marketing Assistant Human Resources @ MR-MRI St. Charles", 
                                      ])
    return recommender_selectbox

# Loading the datasets before implementing it in streamlit
jobs, open_positions, companies_hiring, top_5_companies, top_5_positions, jobs_pivot, positions_pivot = load_data_jobs()
user_view, le_applicant, le_title, matrix, svd, matrix_applicant, matrix_title, cosine_distance_matrix_title, active_jobhunters, avg_job_posts_viewed_per_jobhunter, avg_viewtime_per_job_post, imputer = load_data_user_view()
data_sml_x, y, rf_pickle, ordinal_pickle, label_pickle = load_data_sml()
# user_exp = load_data_user_exp()
# user_int = load_data_user_int()
recommender_selectbox = load_recommender_selectbox()


# Defining the function for the recommender system
def similar_title(title, n):
    """
    this function performs similarity search based on titles
    title: jobs title (str)
    n: number of recommendations
    """
    ix = le_title.transform([title])[0]
    sim_title = le_title.inverse_transform(np.argsort(cosine_distance_matrix_title[ix,:])[:n+1])
    return sim_title[1:]


# Streamlit deployment

# Defining a title that appears at the top of the page
st.write('# Welcome to JobHunter 🔍')


# Creating tabs to navigate through the app
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Welcome', 'Find jobs by location', 'Find similar jobs', 'Predict job category', 'Facts about the job market', 'Tips and tricks for JobHunters'])



# Within tab 1, an input line is created where users can write their name. The name is then included in a text that introduces the app to the user
with tab1:
    st.markdown('#') # used to create spacing between the tabs and the name input
    name = st.text_input('Please enter your name and hit submit to get started on JobHunter!')

# Once the user hits submit, the text below appears on the screen
    if st.button('Submit'):
        st.write(' ')
        st.write('### Hi', name, '! 👋')
        st.write(' ')
        st.markdown(
             """
            Welcome to JobHunter, AAU's newest job searching platform for young professionals!
            - Are you looking for jobs in a specific area? Then browse through the map to find open positions in areas of your choice! 🌎 
            - Have you found an interesting job but wish to see similar opportunities? Then check out the tab 'Find similar jobs' to get recommendations! 📊 
            - Are you curious which job category your attributes match? Try the predictor to determine which job category suits you best! 👤
            - Do you want to learn more about the job market in general? Then head to the fact page to spot numbers and trends! 📈
            - Have you found your dream job and are you ready to apply? Then make sure to use our tips and tricks to write a powerful CV and ace the interview! 📝
            """
        )


# Within tab2, a selectbox box is created so the user can interact with the map by selecting states. The map thereby shows positions in the selected state(s)
with tab2:

    # Multiselector based on states from the jobs dataset
    select_state = st.multiselect('Select state', jobs.groupby('State.Name').count().reset_index()['State.Name'].tolist())

    # If state = 0 (i.e. no inputs from the user), the map shows all jobs available. Else it will show only jobs in the selected states
    if len(select_state) == 0:
            new_var = jobs
    else:
            new_var = jobs[jobs['State.Name'].isin(select_state)]

    # Defining the pydeck map
    layer = pdk.Layer(
            "ScatterplotLayer",
            data=new_var[['Company','Position','State.Name', 'City', 'Employment.Type', "Longitude", "Latitude"]],
            pickable=True,
            opacity=0.7,
            stroked=True,
            filled=True,
            radius_scale=10,
            radius_min_pixels=1,
            radius_max_pixels=100,
            line_width_min_pixels=1,
            get_position=["Longitude", "Latitude"],
            get_radius=1000,
            get_color=[255, 140, 0],
            get_line_color=[0, 0, 0],
        )

    # Setting the viewport location
    view_state = pdk.ViewState(latitude=jobs['Latitude'].mean(), longitude=jobs['Longitude'].mean(), zoom=3.5, pitch=40)

    # Defining the renders
    jobs_map = pdk.Deck(layers=[layer], 
    initial_view_state=view_state,

    # Defining the tooltip when hovering over points in the map
    tooltip={"text": "Company: {Company}\nPosition: {Position}\n Employment type: {Employment.Type}"})

    # Inserting the map in the streamlit app
    st.pydeck_chart(jobs_map)



# Within tab3, a UML recommender system is added together with a selectbox and slider
with tab3:

    # Defining the values in the selectbox and creating a slider to choose between 1-5 recommendations
    select_title = st.selectbox('Select job title', recommender_selectbox)
    n_recs = st.slider('How many recommendations?', 1, 5, 3)

    # If the user presses the button, the UML recommender function 'similar_title' runs based on the selected title and selected recommendations
    if st.button('Show me recommended jobs'):
        st.write(similar_title(select_title, n_recs), use_container_width=True)

# Within tab4, SML
with tab4:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        state = st.selectbox('Select state', options=ordinal_pickle.categories_[0])
    
    with col2:
        employment_type = st.selectbox('Select employment type', options=ordinal_pickle.categories_[1])

    with col3:
        position_name = st.selectbox('Select position', ('Founder/CEO', 'Summer Associate', 'Program leader', 'cashier/bartender', 'Sr. Strategist, Commerce Ops', 'Sales Associate'))
    
    with col4:
        position_interest = st.selectbox('Select interest', ('Host', 'Server', 'Bartender', 'Line Cook', 'Customer Service Rep', 'Receptionist', 'Book Keeper', 'Sales Rep'))

    if st.button("Predict"):
        prediction = rf_pickle.predict(data_sml_x[['State.Name', 'Employment.Type', 'Position.Name_1', 'Position.Of.Interest_1']].iloc[:1, :])
        if prediction == 0:
            st.write("Administrative Assistant ")
        elif prediction == 1: 
            st.write("Executive / Personal Assistant")
        elif prediction == 2: 
            st.write("Customer Service")
        elif prediction == 3: 
            st.write("Accounting and Finance")
        elif prediction == 4: 
            st.write("Receptionist")
        elif prediction == 5: 
            st.write("Retail Professional")
        elif  prediction == 6: 
            st.write("Restaurant")
        elif prediction == 7: 
            st.write("Security Officer")
        elif prediction == 8: 
            st.write("Driver Logistics specialist")


       #result = prediction(state, employment_type, position_name, position_interest) 
        #st.success('Your area of work should be within: {}'.format(result))
        #print(result)




# Within tab5, the first row is divided into 5 columns showing various metrics
# The second row is divided into two columns, showing bar charts of the top 5 companies hiring and top 5 positions hiring
# Next follows two timelines showing the number of jobs posted on a given date, interactive in terms of employment type and position respectively
with tab5:

    st.markdown('#')
    
    # Showing various metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Open positions", open_positions, "+3%")
    col2.metric("Companies hiring", companies_hiring, "-1%")
    col3.metric("Active JobHunters", active_jobhunters, "+8%")
    col4.metric("Avg. job posts viewed per JobHunter", avg_job_posts_viewed_per_jobhunter, "-4%")
    col5.metric("Avg. viewtime per job post (min.)", avg_viewtime_per_job_post, "+7%")

    st.markdown('#')

    # Showing the top 5 companies hiring and the top 5 positions hiring
    col1, col2 = st.columns(2)

    with col1:
        st.write('#### Top 5 companies hiring right now')
        st.bar_chart(top_5_companies, use_container_width=True)
        

    with col2:
        st.write('#### Top 5 professions hiring right now')
        st.bar_chart(top_5_positions, use_container_width=True)

    st.write('#### Timeline trends')
    st.write('The number of jobs posted change throughout the year, check out the trends below based on employment type or position')
    # Inserting a multiselectbox based on employment type and showing the timeline as a plotly chart
    y_axis_val = st.multiselect('Select employment type', options=jobs_pivot.columns)
    jobs_pivot_plot = px.line(jobs_pivot, y=y_axis_val)
    st.plotly_chart(jobs_pivot_plot, use_container_width=True)

    # Inserting a multiselectbox based on position and showing the timeline as a plotly chart
    y_axis_val = st.multiselect('Select position', options=positions_pivot.columns)
    positions_pivot_plot = px.line(positions_pivot, y=y_axis_val)
    st.plotly_chart(positions_pivot_plot, use_container_width=True)



# Within tab6, each row is divided into 3 columns. Each columns includes a title and links to youtube videos, CV templates and various links respectively
with tab6:

    # A header is created for the row. Each column in the row includes a link to a CV youtube video
    st.header('Tutorials ')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('##### How to write a powerful CV')
        st.video("https://www.youtube.com/watch?v=uG2aEh5xBJE")

    with col2:
        st.write('##### How to write a powerful cover letter')
        st.video("https://www.youtube.com/watch?v=lq6aGl1QBRs")
    
    with col3:
        st.write('##### How to prepare for a job interview')
        st.video("https://www.youtube.com/watch?v=enD8mK9Zvwo&list=RDCMUCIEU-iRzjXYo8JrOT9WGpnw&index=2")


    # A header is created for the row. Each column in the row includes a link to a CV template
    st.header('CV templates ')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('https://www.cvmaker.dk/assets/images/cvs/2/cv-example-harvard-3f6591.jpg')

    with col2:
        st.image('https://www.cvmaker.dk/assets/images/cvs/9/cv-example-edinburgh-505577.jpg')

    with col3:
        st.image('https://www.cvmaker.dk/assets/images/cvs/4/cv-example-cambridge-3f6592.jpg')


    # A header is created for the row. Each column in the row includes links to various tests and articles.
    st.header('Want to learn more?')

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.expander('Test Yourself'):
            st.write('- Test your personality type via this [link](https://www.jobindex.dk/persontypetest?lang=en)')
            st.write('- Test your job satisfaction via this [link](https://www.jobindex.dk/test-dig-selv/jobglaede?lang=en)')
            st.write('- Test your stress level via this [link](https://www.jobindex.dk/stress?lang=en)')
            st.write('- Test your talents via this [link](https://www.jobindex.dk/talenttest?lang=en)')
            st.write('- Test your salary via this [link](https://www.jobindex.dk/tjek-din-loen?lang=en)')

    with col2:
        with st.expander('Career development'):
            st.write('- Learn about career development via this [link](https://www.thebalancemoney.com/what-is-career-development-525496)')
            st.write('- Tips to improve career development via this [link](https://www.thebalancemoney.com/improving-career-development-4058289)')
            st.write('- Examine the benefits of mentoring via this [link](https://www.thebalancemoney.com/use-mentoring-to-develop-employees-1918189)')
            st.write('- Explore the concept of job-shadowing via this [link](https://www.thebalancemoney.com/job-shadowing-is-effective-on-the-job-training-1919285)')

    with col3:
        with st.expander('Guidance'):
            st.write('- Free career guidance via this [link](https://www.jobindex.dk/cms/jobvejledning?lang=en)')
            st.write('- Courses for unemployed via this [link](https://jobindexkurser.dk/kategori/alle-kategorier?lang=en)')
