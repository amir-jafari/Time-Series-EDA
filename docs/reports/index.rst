.. _reports:

Sample EDA Reports
==================

These are tseda-generated HTML EDA reports from the
:doc:`Global Air Pollution notebook <../examples/index>`.
Each report covers one air-quality indicator across 175 countries.

Click any link to open the full interactive report in a new tab.

.. raw:: html

   <style>
   .report-grid {
       display: grid;
       grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
       gap: 16px;
       margin: 1.5em 0;
   }
   .report-card {
       border: 1px solid #e1e4e8;
       border-radius: 6px;
       padding: 16px 20px;
       background: #f8f9fa;
       transition: box-shadow .15s;
   }
   .report-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.1); }
   .report-card h3 { margin: 0 0 6px; font-size: 1rem; color: #2c3e50; }
   .report-card p  { margin: 0 0 12px; font-size: .88rem; color: #555; }
   .report-card a.btn {
       display: inline-block;
       background: #2980b9;
       color: #fff !important;
       padding: 5px 14px;
       border-radius: 4px;
       font-size: .85rem;
       text-decoration: none !important;
   }
   .report-card a.btn:hover { background: #3498db; }
   </style>

   <div class="report-grid">

     <div class="report-card">
       <h3>AQI — Overall</h3>
       <p>Composite air-quality index across all pollutants.</p>
       <a class="btn" href="../_static/reports/AQI.html" target="_blank">Open Report ↗</a>
     </div>

     <div class="report-card">
       <h3>PM2.5 AQI</h3>
       <p>Fine particulate matter — the leading air-quality health risk.</p>
       <a class="btn" href="../_static/reports/PM25_AQI.html" target="_blank">Open Report ↗</a>
     </div>

     <div class="report-card">
       <h3>Ozone AQI</h3>
       <p>Ground-level ozone formed by sunlight reacting with pollutants.</p>
       <a class="btn" href="../_static/reports/Ozone_AQI.html" target="_blank">Open Report ↗</a>
     </div>

     <div class="report-card">
       <h3>NO₂ AQI</h3>
       <p>Nitrogen dioxide — primarily from vehicle and industrial emissions.</p>
       <a class="btn" href="../_static/reports/NO2_AQI.html" target="_blank">Open Report ↗</a>
     </div>

     <div class="report-card">
       <h3>CO AQI</h3>
       <p>Carbon monoxide — an indicator of incomplete combustion.</p>
       <a class="btn" href="../_static/reports/CO_AQI.html" target="_blank">Open Report ↗</a>
     </div>

   </div>