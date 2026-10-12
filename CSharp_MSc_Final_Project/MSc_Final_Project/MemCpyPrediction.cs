using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Data.OleDb;
using System.Runtime.InteropServices;
using Microsoft.Office.Interop.Excel;

namespace MSc_Final_Project
{
    public partial class MemCpyPrediction : Form
    {
        public static String CpyFileConStr;
        public static OleDbConnection CpyFileOleDbCon;
        public int HtoD_Cnt, DtoH_Cnt;
        public MemCpyPrediction()
        {
            InitializeComponent();
        }

        private void SelectActualMemoryCpyFile_Btn_Click(object sender, EventArgs e)
        {
            CpyFile_openFileDialog.Filter = "Excel 97-2003 |*.xls|" + "Excel Workbook |*.xlsx";
            CpyFile_openFileDialog.FilterIndex = 2;
            CpyFile_openFileDialog.FileName = "";
            DialogResult DR = CpyFile_openFileDialog.ShowDialog();
            if (DR == DialogResult.OK)
            {
                String Path = CpyFile_openFileDialog.FileName;
                SelectCpyFile_TxtBox.Text = Path;
                CpyFileConStr = @"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=" + Path + ";Extended Properties=Excel 12.0";
                CpyFileOleDbCon = new OleDbConnection(CpyFileConStr);
                CpyFileOleDbCon.Open();
                Update_ComboBox();
                CpyFileOleDbCon.Close();
            }
        }

        private void CpyFileWorksheets_ComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void LoadActualMemoryCpyFile_Btn_Click(object sender, EventArgs e)
        {
            CpyFileOleDbCon.Open();
            OleDbCommand OleDbcmd = new OleDbCommand("Select * From [" + CpyFileWorksheets_ComboBox.GetItemText(CpyFileWorksheets_ComboBox.SelectedItem) + "$]", CpyFileOleDbCon);
            System.Data.DataTable DT = new System.Data.DataTable();
            DT.Load(OleDbcmd.ExecuteReader());
            MemCpy_GridView.DataSource = DT;
            CpyFileOleDbCon.Close();
        }
        private void Update_ComboBox()
        {
            CpyFileWorksheets_ComboBox.Items.Clear();
            CpyFileWorksheets_ComboBox.ResetText();
            System.Data.DataTable Sheet_DT = new System.Data.DataTable();
            Sheet_DT = CpyFileOleDbCon.GetOleDbSchemaTable(OleDbSchemaGuid.Tables, null);
            foreach (DataRow row in Sheet_DT.Rows)
            {
                CpyFileWorksheets_ComboBox.Items.Add(row["TABLE_NAME"].ToString().Replace("$", string.Empty));
            }
        }
        private void RemoveNonCpyRows_Btn_Click(object sender, EventArgs e)
        {
            for (int i = MemCpy_GridView.Rows.Count - 1; i >= 0; i--)
            {
                if (((MemCpy_GridView.Rows[i].Cells[0].Value.ToString() != "Memcpy HtoD [async]") && (MemCpy_GridView.Rows[i].Cells[0].Value.ToString() != "Memcpy DtoH [async]")) || (Convert.ToDouble(MemCpy_GridView.Rows[i].Cells["Size(bytes)"].Value) < 5000000))
                {
                    MemCpy_GridView.Rows.RemoveAt(i);
                }
            }
            //foreach(DataGridViewRow row in MemCpy_GridView.Rows)
            //{
            //    try
            //    {
            //        if (((row.Cells[0].Value.ToString() != "Memcpy HtoD") && (row.Cells[0].Value.ToString() != "Memcpy DtoH")) || (Convert.ToDouble(row.Cells["Size(bytes)"].Value) < 5000000))
            //        {
            //            MemCpy_GridView.Rows.RemoveAt(row.Index);
            //        }
            //    }
            //    catch
            //    {
            //        try
            //        {
            //            MemCpy_GridView.Rows.RemoveAt(row.Index);
            //        }
            //        catch { }
            //    }
            //}
        }
        private void CpyAvgError_Btn_Click(object sender, EventArgs e)
        {
            HtoD_Cnt = 0;
            DtoH_Cnt = 0;
            double HtoDAvg = 0;
            double DtoHAvg = 0;
            //double HtoD_Error = 0;
            //double DtoH_Error = 0;
            //double Actual_HtoD, Predict_HtoD, Actual_DtoH, Predict_DtoH;
            foreach (DataGridViewRow row in MemCpy_GridView.Rows)
            {
                if (row.Cells["Name"].Value.ToString() == "Memcpy HtoD [async]")
                {
                    HtoDAvg += Convert.ToDouble(row.Cells["Throughput(bytes/sec)"].Value);
                    //Actual_HtoD = (Convert.ToDouble(row.Cells["Size(bytes)"].Value) / Convert.ToDouble(row.Cells["Throughput(bytes/sec)"].Value)) * 1000;
                    //Predict_HtoD = (Convert.ToDouble(row.Cells["Size(bytes)"].Value) / CoScheduler.H2DAsynccpyBW) * 1000;
                    //HtoD_Error += Math.Abs(((Actual_HtoD - Predict_HtoD) / Actual_HtoD) * 100);
                    HtoD_Cnt++;
                }
                if (row.Cells["Name"].Value.ToString() == "Memcpy DtoH [async]")
                {
                    DtoHAvg += Convert.ToDouble(row.Cells["Throughput(bytes/sec)"].Value);
                    //Actual_DtoH = (Convert.ToDouble(row.Cells["Size(bytes)"].Value) / Convert.ToDouble(row.Cells["Throughput(bytes/sec)"].Value)) * 1000;
                    //Predict_DtoH = (Convert.ToDouble(row.Cells["Size(bytes)"].Value) / CoScheduler.D2HAsynccpyBW) * 1000;
                    //DtoH_Error += Math.Abs(((Actual_DtoH - Predict_DtoH) / Actual_DtoH) * 100);
                    DtoH_Cnt++;
                }
            }
            HtoDAvg = HtoDAvg / HtoD_Cnt;
            DtoHAvg = DtoHAvg / DtoH_Cnt;
            ActualHtoDAvg_TxtBox.Text = (Convert.ToInt64(HtoDAvg)).ToString();
            ActualDtoHAvg_TxtBox.Text = (Convert.ToInt64(DtoHAvg)).ToString();
            HtoDAvg_TxtBox.Text = Math.Round((Math.Abs(((HtoDAvg - CoScheduler.H2DAsynccpyBW) / HtoDAvg) * 100)), 2).ToString();
            DtoHAvg_TxtBox.Text = Math.Round((Math.Abs(((DtoHAvg - CoScheduler.D2HAsynccpyBW) / DtoHAvg) * 100)), 2).ToString();
            //HtoDAvg_TxtBox.Text = (HtoD_Error / HtoD_Cnt).ToString();
            //DtoHAvg_TxtBox.Text = (DtoH_Error / DtoH_Cnt).ToString();
        }

    }
}
