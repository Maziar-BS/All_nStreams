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
using System.Data.Sql;
using System.Data.SqlClient;

namespace MSc_Final_Project
{
    public partial class SchedulerPrediction : Form
    {
        public static String ExeSetConStr;
        public static OleDbConnection ExeSetOleDbCon;
        public static int[] Combs = { 0, 0, 0, 0 };
        public SchedulerPrediction()
        {
            InitializeComponent();
        }

        private void SchedulerPrediction_Load(object sender, EventArgs e)
        {
            LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
            var CombinationSet = DB.CombinationsTables.Where(P => P.ID == 1).Select(P => P).Single();
            Combs[0] = CombinationSet.C1;
            Combs[1] = CombinationSet.C2;
            Combs[2] = CombinationSet.C3;
            Combs[3] = CombinationSet.C4;
            PES_GridView.ColumnCount = CoScheduler.NumofStreams + 1;
            for (int i = 0; i < CoScheduler.NumofStreams; i++)
            {
                PES_GridView.Columns[i].Name = "S" + (i + 1).ToString();
            }
            PES_GridView.Columns[CoScheduler.NumofStreams].Name = "Duration";
            AES_GridView.ColumnCount = CoScheduler.NumofStreams + 1;
            for (int i = 0; i < CoScheduler.NumofStreams; i++)
            {
                AES_GridView.Columns[i].Name = "S" + (i + 1).ToString();
            }
            AES_GridView.Columns[CoScheduler.NumofStreams].Name = "Duration";
            Fill_GridViews();
            ImportnonStreamSetFile_Btn.Enabled = false;
            ImportPredictedFile_Btn.Enabled = false;
            ImportActual2File_Btn.Enabled = false;
            SchedulerExeSet_Btn.Enabled = false;
            ActualExeSet_Btn.Enabled = false;
        }

        private void Fill_GridViews()
        {
            int row = 0;
            for (int i = 1; i <= CoScheduler.InputPrograms; i++)
            {
                for (int j = 1; j <= CoScheduler.InputPrograms; j++)
                {
                    for (int k = 1; k <= CoScheduler.InputPrograms; k++)
                    {
                        if ((i != j) && (i != k) && (j != k))
                        {
                            PES_GridView.Rows.Add();
                            PES_GridView.Rows[row].Cells[0].Value = i;
                            PES_GridView.Rows[row].Cells[1].Value = j;
                            PES_GridView.Rows[row].Cells[2].Value = k;
                            AES_GridView.Rows.Add();
                            AES_GridView.Rows[row].Cells[0].Value = i;
                            AES_GridView.Rows[row].Cells[1].Value = j;
                            AES_GridView.Rows[row].Cells[2].Value = k;
                            row++;
                        }
                    }
                }
            }
        }

        private void SelectnonStreamSetFile_Btn_Click(object sender, EventArgs e)
        {
            ExeSetopenFileDialog.Filter = "Excel 97-2003 |*.xls|" + "Excel Workbook |*.xlsx";
            ExeSetopenFileDialog.FilterIndex = 2;
            ExeSetopenFileDialog.FileName = "";
            DialogResult DR = ExeSetopenFileDialog.ShowDialog();
            if (DR == DialogResult.OK)
            {
                String Path = ExeSetopenFileDialog.FileName;
                nonStreamSetFileLocation_TxtBox.Text = Path;
                ExeSetConStr = @"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=" + Path + ";Extended Properties=Excel 12.0";
                ExeSetOleDbCon = new OleDbConnection(ExeSetConStr);
                ExeSetOleDbCon.Open();
                ExeSetUpdate_ComboBox(nonStreamWorksheets_ComboBox);
                ExeSetOleDbCon.Close();
            }
            ImportnonStreamSetFile_Btn.Enabled = false;
        }

        private void nonStreamWorksheets_ComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            ImportnonStreamSetFile_Btn.Enabled = true;
        }

        private void ImportnonStreamSetFile_Btn_Click(object sender, EventArgs e)
        {
            ExeSetOleDbCon.Open();
            OleDbCommand OleDbcmd = new OleDbCommand("Select * From [" + nonStreamWorksheets_ComboBox.GetItemText(nonStreamWorksheets_ComboBox.SelectedItem) + "$]", ExeSetOleDbCon);
            System.Data.DataTable DT = new System.Data.DataTable();
            DT.Load(OleDbcmd.ExecuteReader());
            nonStream_GridView.DataSource = DT;
            ExeSetOleDbCon.Close();
            ImportnonStreamSetFile_Btn.Enabled = false;
        }
        private void SelectPredictedStreamSetFile_Btn_Click(object sender, EventArgs e)
        {
            ExeSetopenFileDialog.Filter = "Excel 97-2003 |*.xls|" + "Excel Workbook |*.xlsx";
            ExeSetopenFileDialog.FilterIndex = 2;
            ExeSetopenFileDialog.FileName = "";
            DialogResult DR = ExeSetopenFileDialog.ShowDialog();
            if (DR == DialogResult.OK)
            {
                String Path = ExeSetopenFileDialog.FileName;
                PredictedFileLocation_TxtBox.Text = Path;
                ExeSetConStr = @"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=" + Path + ";Extended Properties=Excel 12.0";
                ExeSetOleDbCon = new OleDbConnection(ExeSetConStr);
                ExeSetOleDbCon.Open();
                ExeSetUpdate_ComboBox(PredictedFileWorksheet_ComboBox);
                ExeSetOleDbCon.Close();
            }
            ImportPredictedFile_Btn.Enabled = false;
            SchedulerExeSet_Btn.Enabled = false;
        }

        private void PredictedFileWorksheet_ComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            ImportPredictedFile_Btn.Enabled = true;
            SchedulerExeSet_Btn.Enabled = false;
        }

        private void ImportPredictedFile_Btn_Click(object sender, EventArgs e)
        {
            ExeSetOleDbCon.Open();
            OleDbCommand OleDbcmd = new OleDbCommand("Select * From [" + PredictedFileWorksheet_ComboBox.GetItemText(PredictedFileWorksheet_ComboBox.SelectedItem) + "$]", ExeSetOleDbCon);
            System.Data.DataTable DT = new System.Data.DataTable();
            DT.Load(OleDbcmd.ExecuteReader());
            PredictedStreamSetFile_GridView.DataSource = DT;
            ExeSetOleDbCon.Close();
            foreach (DataGridViewRow row in PredictedStreamSetFile_GridView.Rows)
            {
                try
                {
                    //PES_GridView.Rows[row.Index].Cells[CoScheduler.NumofStreams].Value = Math.Round((Convert.ToDouble(nonStream_GridView.Rows[row.Index].Cells[1].Value) - Convert.ToDouble(row.Cells[1].Value)), 2);
                    PES_GridView.Rows[row.Index].Cells[CoScheduler.NumofStreams].Value = Math.Round(Convert.ToDouble(row.Cells[1].Value), 2);
                }
                catch { }
            }
            ImportPredictedFile_Btn.Enabled = false;
            SchedulerExeSet_Btn.Enabled = true;
        }

        private void SelectActualStreamSetFile_Btn_Click(object sender, EventArgs e)
        {
            ExeSetopenFileDialog.Filter = "Excel 97-2003 |*.xls|" + "Excel Workbook |*.xlsx";
            ExeSetopenFileDialog.FilterIndex = 2;
            ExeSetopenFileDialog.FileName = "";
            DialogResult DR = ExeSetopenFileDialog.ShowDialog();
            if (DR == DialogResult.OK)
            {
                String Path = ExeSetopenFileDialog.FileName;
                Actual2FileLocation_TxtBox.Text = Path;
                ExeSetConStr = @"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=" + Path + ";Extended Properties=Excel 12.0";
                ExeSetOleDbCon = new OleDbConnection(ExeSetConStr);
                ExeSetOleDbCon.Open();
                ExeSetUpdate_ComboBox(ActualFileWorksheet2_ComboBox);
                ExeSetOleDbCon.Close();
            }
            ImportActual2File_Btn.Enabled = false;
            ActualExeSet_Btn.Enabled = false;
        }

        private void ActualFileWorksheet_ComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            ImportActual2File_Btn.Enabled = true;
            ActualExeSet_Btn.Enabled = false;
        }
        private void ImportActualFile_Btn_Click(object sender, EventArgs e)
        {
            ExeSetOleDbCon.Open();
            OleDbCommand OleDbcmd = new OleDbCommand("Select * From [" + ActualFileWorksheet2_ComboBox.GetItemText(ActualFileWorksheet2_ComboBox.SelectedItem) + "$]", ExeSetOleDbCon);
            System.Data.DataTable DT = new System.Data.DataTable();
            DT.Load(OleDbcmd.ExecuteReader());
            Actual2StreamSetFile_GridView.DataSource = DT;
            ExeSetOleDbCon.Close();
            foreach (DataGridViewRow row in Actual2StreamSetFile_GridView.Rows)
            {
                try
                {
                    //AES_GridView.Rows[row.Index].Cells[CoScheduler.NumofStreams].Value = Math.Round((Convert.ToDouble(nonStream_GridView.Rows[row.Index].Cells[1].Value) - Convert.ToDouble(row.Cells[1].Value)), 2);
                    AES_GridView.Rows[row.Index].Cells[CoScheduler.NumofStreams].Value = Math.Round(Convert.ToDouble(row.Cells[1].Value), 2);
                }
                catch { }
            }
            ActualExeSet_Btn.Enabled = true;
            ImportActual2File_Btn.Enabled = false;
        }

        private void ExeSetUpdate_ComboBox(ComboBox CB)
        {
            CB.Items.Clear();
            CB.ResetText();
            System.Data.DataTable Sheet_DT = new System.Data.DataTable();
            Sheet_DT = ExeSetOleDbCon.GetOleDbSchemaTable(OleDbSchemaGuid.Tables, null);
            foreach (DataRow row in Sheet_DT.Rows)
            {
                CB.Items.Add(row["TABLE_NAME"].ToString().Replace("$", string.Empty));
            }
        }

        private void SchedulerExeSet_Btn_Click(object sender, EventArgs e)
        {
            //LinqDataContext DB = new LinqDataContext();
            //List<int[]> ExeSet = new List<int[]>();
            //List<double> OptExeSet = new List<double>();
            //int PID1;
            //IIIStreamSetTable First, Second;
            //ExeSetGenerator(ExeSet);
            //for (int i = 0; i < ExeSet.Count(); i++)
            //{
            //    PID1 = DB.IIStreamSetTables.Where(P => (P.PID1 == ExeSet[i][0]) && (P.PID2 == ExeSet[i][1])).Select(P => P.ID).Single();
            //    First = DB.IIIStreamSetTables.Where(P => (P.PID1 == PID1) && (P.PID2 == ExeSet[i][2])).Select(P => P).Single();
            //    PID1 = DB.IIStreamSetTables.Where(P => (P.PID1 == ExeSet[i][3]) && (P.PID2 == ExeSet[i][4])).Select(P => P.ID).Single();
            //    Second = DB.IIIStreamSetTables.Where(P => (P.PID1 == PID1) && (P.PID2 == ExeSet[i][5])).Select(P => P).Single();
            //    OptExeSet.Add(First.MemcpyH2D + First.Kernel + First.MemcpyD2H + Second.MemcpyH2D + Second.Kernel + Second.MemcpyD2H);
            //}
            //double min = 1000000000;
            //int ExeSetNum = 0;
            //for (int i = 0; i < OptExeSet.Count(); i++)
            //{
            //    if (OptExeSet[i] < min)
            //    {
            //        min = OptExeSet[i];
            //        ExeSetNum = i;
            //    }
            //}
            //PO_TxtBox.Text = "Optimum ExeSet Duration = " + min.ToString() + " || Optimum ExeSet includes: P" + (ExeSet[ExeSetNum][0]).ToString() + "_P" + (ExeSet[ExeSetNum][1]).ToString() + "_P" + (ExeSet[ExeSetNum][2]).ToString() + " | P" + (ExeSet[ExeSetNum][3]).ToString() + "_P" + (ExeSet[ExeSetNum][4]).ToString() + "_P" + (ExeSet[ExeSetNum][5]).ToString();
            double ExeSetDuration = 1000000000;
            double MaxExeSetDuration = 0;
            int[] ExeSetOrder = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            int[] MaxExeSetOrder = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            if (CoScheduler.InputPrograms == 6)
            {
                for (int i = 0; i < NumofStreamSets(); i++)
                {
                    for (int j = 0; j < NumofStreamSets(); j++)
                    {
                        if (Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value))
                        {
                            if (Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value))
                            {
                                if (Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value))
                                {
                                    if (Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value))
                                    {
                                        if (Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value))
                                        {
                                            if (Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value))
                                            {
                                                if (Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value))
                                                {
                                                    if (Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value))
                                                    {
                                                        if (Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value))
                                                        {
                                                            if (ExeSetDuration > (Convert.ToDouble(PES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[j].Cells[3].Value)))
                                                            {
                                                                ExeSetDuration = Convert.ToDouble(PES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[j].Cells[3].Value);
                                                                ExeSetOrder[0] = Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value);
                                                                ExeSetOrder[1] = Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value);
                                                                ExeSetOrder[2] = Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value);
                                                                ExeSetOrder[3] = Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value);
                                                                ExeSetOrder[4] = Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value);
                                                                ExeSetOrder[5] = Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value);
                                                            }
                                                            if (MaxExeSetDuration < (Convert.ToDouble(PES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[j].Cells[3].Value)))
                                                            {
                                                                MaxExeSetDuration = Convert.ToDouble(PES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[j].Cells[3].Value);
                                                                MaxExeSetOrder[0] = Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value);
                                                                MaxExeSetOrder[1] = Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value);
                                                                MaxExeSetOrder[2] = Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value);
                                                                MaxExeSetOrder[3] = Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value);
                                                                MaxExeSetOrder[4] = Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value);
                                                                MaxExeSetOrder[5] = Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (CoScheduler.InputPrograms == 9)
            {
                for (int i = 0; i < NumofStreamSets(); i++)
                {
                    for (int j = 0; j < NumofStreamSets(); j++)
                    {
                        for (int k = 0; k < NumofStreamSets(); k++)
                        {
                            if (Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value))
                            {
                                if (Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value))
                                {
                                    if (Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value))
                                    {
                                        if (Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[0].Value))
                                        {
                                            if (Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[1].Value))
                                            {
                                                if (Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[2].Value))
                                                {
                                                    if (Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value))
                                                    {
                                                        if (Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value))
                                                        {
                                                            if (Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value))
                                                            {
                                                                if (Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[0].Value))
                                                                {
                                                                    if (Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[1].Value))
                                                                    {
                                                                        if (Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[2].Value))
                                                                        {
                                                                            if (Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value))
                                                                            {
                                                                                if (Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value))
                                                                                {
                                                                                    if (Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value))
                                                                                    {
                                                                                        if (Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[0].Value))
                                                                                        {
                                                                                            if (Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[1].Value))
                                                                                            {
                                                                                                if (Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[2].Value))
                                                                                                {
                                                                                                    if (Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[0].Value))
                                                                                                    {
                                                                                                        if (Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[1].Value))
                                                                                                        {
                                                                                                            if (Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[2].Value))
                                                                                                            {
                                                                                                                if (Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[0].Value))
                                                                                                                {
                                                                                                                    if (Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[1].Value))
                                                                                                                    {
                                                                                                                        if (Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[2].Value))
                                                                                                                        {
                                                                                                                            if (Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[0].Value))
                                                                                                                            {
                                                                                                                                if (Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[1].Value))
                                                                                                                                {
                                                                                                                                    if (Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value) != Convert.ToInt32(PES_GridView.Rows[k].Cells[2].Value))
                                                                                                                                    {
                                                                                                                                        if (ExeSetDuration > (Convert.ToDouble(PES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[j].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[k].Cells[3].Value)))
                                                                                                                                        {
                                                                                                                                            ExeSetDuration = Convert.ToDouble(PES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[j].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[k].Cells[3].Value);
                                                                                                                                            ExeSetOrder[0] = Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value);
                                                                                                                                            ExeSetOrder[1] = Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value);
                                                                                                                                            ExeSetOrder[2] = Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value);
                                                                                                                                            ExeSetOrder[3] = Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value);
                                                                                                                                            ExeSetOrder[4] = Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value);
                                                                                                                                            ExeSetOrder[5] = Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value);
                                                                                                                                            ExeSetOrder[6] = Convert.ToInt32(PES_GridView.Rows[k].Cells[0].Value);
                                                                                                                                            ExeSetOrder[7] = Convert.ToInt32(PES_GridView.Rows[k].Cells[1].Value);
                                                                                                                                            ExeSetOrder[8] = Convert.ToInt32(PES_GridView.Rows[k].Cells[2].Value);
                                                                                                                                        }
                                                                                                                                        if (MaxExeSetDuration < (Convert.ToDouble(PES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[j].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[k].Cells[3].Value)))
                                                                                                                                        {
                                                                                                                                            MaxExeSetDuration = Convert.ToDouble(PES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[j].Cells[3].Value) + Convert.ToDouble(PES_GridView.Rows[k].Cells[3].Value);
                                                                                                                                            MaxExeSetOrder[0] = Convert.ToInt32(PES_GridView.Rows[i].Cells[0].Value);
                                                                                                                                            MaxExeSetOrder[1] = Convert.ToInt32(PES_GridView.Rows[i].Cells[1].Value);
                                                                                                                                            MaxExeSetOrder[2] = Convert.ToInt32(PES_GridView.Rows[i].Cells[2].Value);
                                                                                                                                            MaxExeSetOrder[3] = Convert.ToInt32(PES_GridView.Rows[j].Cells[0].Value);
                                                                                                                                            MaxExeSetOrder[4] = Convert.ToInt32(PES_GridView.Rows[j].Cells[1].Value);
                                                                                                                                            MaxExeSetOrder[5] = Convert.ToInt32(PES_GridView.Rows[j].Cells[2].Value);
                                                                                                                                            MaxExeSetOrder[6] = Convert.ToInt32(PES_GridView.Rows[k].Cells[0].Value);
                                                                                                                                            MaxExeSetOrder[7] = Convert.ToInt32(PES_GridView.Rows[k].Cells[1].Value);
                                                                                                                                            MaxExeSetOrder[8] = Convert.ToInt32(PES_GridView.Rows[k].Cells[2].Value);
                                                                                                                                        }
                                                                                                                                    }
                                                                                                                                }
                                                                                                                            }
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (CoScheduler.InputPrograms == 6)
            {
                PO_TxtBox.Text = "Predicted Execution Time of Optimum StreamSet = " + ExeSetDuration.ToString() + " || Optimum ExeSet includes: P" + ExeSetOrder[0].ToString() + "_P" + ExeSetOrder[1].ToString() + "_P" + ExeSetOrder[2].ToString() + " | P" + ExeSetOrder[3].ToString() + "_P" + ExeSetOrder[4].ToString() + "_P" + ExeSetOrder[5].ToString();
                Pmax_TxtBox.Text = "Predicted Execution Time of Worst StreamSet = " + MaxExeSetDuration.ToString() + " || Optimum ExeSet includes: P" + MaxExeSetOrder[0].ToString() + "_P" + MaxExeSetOrder[1].ToString() + "_P" + MaxExeSetOrder[2].ToString() + " | P" + MaxExeSetOrder[3].ToString() + "_P" + MaxExeSetOrder[4].ToString() + "_P" + MaxExeSetOrder[5].ToString();
            }
            if (CoScheduler.InputPrograms == 9)
            {
                PO_TxtBox.Text = "Predicted Execution Time of Optimum StreamSet = " + ExeSetDuration.ToString() + " || Optimum ExeSet includes: P" + ExeSetOrder[0].ToString() + "_P" + ExeSetOrder[1].ToString() + "_P" + ExeSetOrder[2].ToString() + " | P" + ExeSetOrder[3].ToString() + "_P" + ExeSetOrder[4].ToString() + "_P" + ExeSetOrder[5].ToString() + " | P" + ExeSetOrder[6].ToString() + "_P" + ExeSetOrder[7].ToString() + "_P" + ExeSetOrder[8].ToString();
                Pmax_TxtBox.Text = "Predicted Execution Time of Worst StreamSet = " + MaxExeSetDuration.ToString() + " || Optimum ExeSet includes: P" + MaxExeSetOrder[0].ToString() + "_P" + MaxExeSetOrder[1].ToString() + "_P" + MaxExeSetOrder[2].ToString() + " | P" + MaxExeSetOrder[3].ToString() + "_P" + MaxExeSetOrder[4].ToString() + "_P" + MaxExeSetOrder[5].ToString() + " | P" + MaxExeSetOrder[6].ToString() + "_P" + MaxExeSetOrder[7].ToString() + "_P" + MaxExeSetOrder[8].ToString();
            }
            SchedulerExeSet_Btn.Enabled = false;
        }

        private void ActualExeSet_Btn_Click(object sender, EventArgs e)
        {
            //LinqDataContext DB = new LinqDataContext();
            ////DB.ExecuteCommand("TRUNCATE TABLE ActualIIIStreamSetTable");
            ////int row = 0;
            ////for (int i = 1; i < CoScheduler.InputPrograms; i++)
            ////{
            ////    for (int j = 1; j <= CoScheduler.InputPrograms; j++)
            ////    {
            ////        for (int k = 1; k <= CoScheduler.InputPrograms; k++)
            ////        {
            ////            if ((i != j) && (i != k) && (j != k))
            ////            {
            ////                DB.SaveActualIIIStreamSetTable(row, Convert.ToDouble(ActualStreamSetFile_GridView.Rows[row].Cells[1].Value), i, j, k);
            ////                row++;
            ////            }
            ////        }
            ////    }
            ////}
            //List<int[]> ExeSet = new List<int[]>();
            //List<double> OptExeSet = new List<double>();
            //int[] tmp = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            ////ActualIIIStreamSetTable First, Second;
            //double First = 0, Second = 0;
            //bool firstisout = false, secondisout = false;
            //ExeSetGenerator(ExeSet);
            //for (int i = 0; i < ExeSet.Count(); i++)
            //{
            //    tmp = ExeSet[i];
            //    foreach (DataGridViewRow row in ActualStreamSetFile_GridView.Rows)
            //    {
            //        if ((firstisout) && (secondisout))
            //        {
            //            OptExeSet.Add(First + Second);
            //            AO_TxtBox.Text = OptExeSet[i].ToString();
            //            firstisout = false;
            //            secondisout = false;
            //            break;
            //        }
            //        try
            //        {
            //            if (row.Cells[0].Value.ToString() == ("S1= " + tmp[0].ToString() + " | S2= " + tmp[1].ToString() + " | S3= " + tmp[2].ToString()))
            //            {
            //                First = Convert.ToDouble(row.Cells[1].Value);
            //                firstisout = true;
            //            }
            //            if (row.Cells[0].Value.ToString() == ("S1= " + tmp[3].ToString() + " | S2= " + tmp[4].ToString() + " | S3= " + tmp[5].ToString()))
            //            {
            //                Second = Convert.ToDouble(row.Cells[1].Value);
            //                secondisout = true;
            //            }
            //        }
            //        catch { }
            //    }
            //    //First = DB.ActualIIIStreamSetTables.Where(P => (P.PID1 == ExeSet[i][0]) && (P.PID2 == ExeSet[i][1]) && (P.PID3 == ExeSet[i][2])).Select(P => P).Single();
            //    //Second = DB.ActualIIIStreamSetTables.Where(P => (P.PID1 == ExeSet[i][3]) && (P.PID2 == ExeSet[i][4]) && (P.PID3 == ExeSet[i][5])).Select(P => P).Single();
            //}
            //double min = 1000000000;
            //int ExeSetNum = 0;
            //for (int i = 0; i < OptExeSet.Count(); i++)
            //{
            //    if (OptExeSet[i] < min)
            //    {
            //        min = OptExeSet[i];
            //        ExeSetNum = i;
            //    }
            //}
            //AO_TxtBox.Text = "Optimum ExeSet Duration = " + min.ToString() + " || Optimum ExeSet includes: P" + (ExeSet[ExeSetNum][0]).ToString() + "_P" + (ExeSet[ExeSetNum][1]).ToString() + "_P" + (ExeSet[ExeSetNum][2]).ToString() + " | P" + (ExeSet[ExeSetNum][3]).ToString() + "_P" + (ExeSet[ExeSetNum][4]).ToString() + "_P" + (ExeSet[ExeSetNum][5]).ToString();
            double ExeSetDuration = 1000000000;
            double MaxExeSetDuration = 0;
            int[] ExeSetOrder = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            int[] MaxExeSetOrder = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            if (CoScheduler.InputPrograms == 6)
            {
                for (int i = 0; i < NumofStreamSets(); i++)
                {
                    for (int j = 0; j < NumofStreamSets(); j++)
                    {
                        if (Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value))
                        {
                            if (Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value))
                            {
                                if (Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value))
                                {
                                    if (Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value))
                                    {
                                        if (Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value))
                                        {
                                            if (Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value))
                                            {
                                                if (Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value))
                                                {
                                                    if (Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value))
                                                    {
                                                        if (Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value))
                                                        {
                                                            if (ExeSetDuration > (Convert.ToDouble(AES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[j].Cells[3].Value)))
                                                            {
                                                                ExeSetDuration = Convert.ToDouble(AES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[j].Cells[3].Value);
                                                                ExeSetOrder[0] = Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value);
                                                                ExeSetOrder[1] = Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value);
                                                                ExeSetOrder[2] = Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value);
                                                                ExeSetOrder[3] = Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value);
                                                                ExeSetOrder[4] = Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value);
                                                                ExeSetOrder[5] = Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value);
                                                            }
                                                            if (MaxExeSetDuration < (Convert.ToDouble(AES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[j].Cells[3].Value)))
                                                            {
                                                                MaxExeSetDuration = Convert.ToDouble(AES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[j].Cells[3].Value);
                                                                MaxExeSetOrder[0] = Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value);
                                                                MaxExeSetOrder[1] = Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value);
                                                                MaxExeSetOrder[2] = Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value);
                                                                MaxExeSetOrder[3] = Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value);
                                                                MaxExeSetOrder[4] = Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value);
                                                                MaxExeSetOrder[5] = Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (CoScheduler.InputPrograms == 9)
            {
                for (int i = 0; i < NumofStreamSets(); i++)
                {
                    for (int j = 0; j < NumofStreamSets(); j++)
                    {
                        for (int k = 0; k < NumofStreamSets(); k++)
                        {
                            if (Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value))
                            {
                                if (Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value))
                                {
                                    if (Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value))
                                    {
                                        if (Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[0].Value))
                                        {
                                            if (Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[1].Value))
                                            {
                                                if (Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[2].Value))
                                                {
                                                    if (Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value))
                                                    {
                                                        if (Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value))
                                                        {
                                                            if (Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value))
                                                            {
                                                                if (Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[0].Value))
                                                                {
                                                                    if (Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[1].Value))
                                                                    {
                                                                        if (Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[2].Value))
                                                                        {
                                                                            if (Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value))
                                                                            {
                                                                                if (Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value))
                                                                                {
                                                                                    if (Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value))
                                                                                    {
                                                                                        if (Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[0].Value))
                                                                                        {
                                                                                            if (Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[1].Value))
                                                                                            {
                                                                                                if (Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[2].Value))
                                                                                                {
                                                                                                    if (Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[0].Value))
                                                                                                    {
                                                                                                        if (Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[1].Value))
                                                                                                        {
                                                                                                            if (Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[2].Value))
                                                                                                            {
                                                                                                                if (Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[0].Value))
                                                                                                                {
                                                                                                                    if (Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[1].Value))
                                                                                                                    {
                                                                                                                        if (Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[2].Value))
                                                                                                                        {
                                                                                                                            if (Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[0].Value))
                                                                                                                            {
                                                                                                                                if (Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[1].Value))
                                                                                                                                {
                                                                                                                                    if (Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value) != Convert.ToInt32(AES_GridView.Rows[k].Cells[2].Value))
                                                                                                                                    {
                                                                                                                                        if (ExeSetDuration > (Convert.ToDouble(AES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[j].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[k].Cells[3].Value)))
                                                                                                                                        {
                                                                                                                                            ExeSetDuration = Convert.ToDouble(AES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[j].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[k].Cells[3].Value);
                                                                                                                                            ExeSetOrder[0] = Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value);
                                                                                                                                            ExeSetOrder[1] = Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value);
                                                                                                                                            ExeSetOrder[2] = Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value);
                                                                                                                                            ExeSetOrder[3] = Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value);
                                                                                                                                            ExeSetOrder[4] = Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value);
                                                                                                                                            ExeSetOrder[5] = Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value);
                                                                                                                                            ExeSetOrder[6] = Convert.ToInt32(AES_GridView.Rows[k].Cells[0].Value);
                                                                                                                                            ExeSetOrder[7] = Convert.ToInt32(AES_GridView.Rows[k].Cells[1].Value);
                                                                                                                                            ExeSetOrder[8] = Convert.ToInt32(AES_GridView.Rows[k].Cells[2].Value);
                                                                                                                                        }
                                                                                                                                        if (MaxExeSetDuration < (Convert.ToDouble(AES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[j].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[k].Cells[3].Value)))
                                                                                                                                        {
                                                                                                                                            MaxExeSetDuration = Convert.ToDouble(AES_GridView.Rows[i].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[j].Cells[3].Value) + Convert.ToDouble(AES_GridView.Rows[k].Cells[3].Value);
                                                                                                                                            MaxExeSetOrder[0] = Convert.ToInt32(AES_GridView.Rows[i].Cells[0].Value);
                                                                                                                                            MaxExeSetOrder[1] = Convert.ToInt32(AES_GridView.Rows[i].Cells[1].Value);
                                                                                                                                            MaxExeSetOrder[2] = Convert.ToInt32(AES_GridView.Rows[i].Cells[2].Value);
                                                                                                                                            MaxExeSetOrder[3] = Convert.ToInt32(AES_GridView.Rows[j].Cells[0].Value);
                                                                                                                                            MaxExeSetOrder[4] = Convert.ToInt32(AES_GridView.Rows[j].Cells[1].Value);
                                                                                                                                            MaxExeSetOrder[5] = Convert.ToInt32(AES_GridView.Rows[j].Cells[2].Value);
                                                                                                                                            MaxExeSetOrder[6] = Convert.ToInt32(AES_GridView.Rows[k].Cells[0].Value);
                                                                                                                                            MaxExeSetOrder[7] = Convert.ToInt32(AES_GridView.Rows[k].Cells[1].Value);
                                                                                                                                            MaxExeSetOrder[8] = Convert.ToInt32(AES_GridView.Rows[k].Cells[2].Value);
                                                                                                                                        }
                                                                                                                                    }
                                                                                                                                }
                                                                                                                            }
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (CoScheduler.InputPrograms == 6)
            {
                AO_TxtBox.Text = "Actual Execution Time of Optimum StreamSet = " + ExeSetDuration.ToString() + " || Optimum ExeSet includes: P" + ExeSetOrder[0].ToString() + "_P" + ExeSetOrder[1].ToString() + "_P" + ExeSetOrder[2].ToString() + " | P" + ExeSetOrder[3].ToString() + "_P" + ExeSetOrder[4].ToString() + "_P" + ExeSetOrder[5].ToString();
                Aman_TxtBox.Text = "Actual Execution Time of Worst StreamSet = " + MaxExeSetDuration.ToString() + " || Optimum ExeSet includes: P" + MaxExeSetOrder[0].ToString() + "_P" + MaxExeSetOrder[1].ToString() + "_P" + MaxExeSetOrder[2].ToString() + " | P" + MaxExeSetOrder[3].ToString() + "_P" + MaxExeSetOrder[4].ToString() + "_P" + MaxExeSetOrder[5].ToString();
            }
            if (CoScheduler.InputPrograms == 9)
            {
                AO_TxtBox.Text = "Actual Execution Time of Optimum StreamSet = " + ExeSetDuration.ToString() + " || Optimum ExeSet includes: P" + ExeSetOrder[0].ToString() + "_P" + ExeSetOrder[1].ToString() + "_P" + ExeSetOrder[2].ToString() + " | P" + ExeSetOrder[3].ToString() + "_P" + ExeSetOrder[4].ToString() + "_P" + ExeSetOrder[5].ToString() + " | P" + ExeSetOrder[6].ToString() + "_P" + ExeSetOrder[7].ToString() + "_P" + ExeSetOrder[8].ToString();
                Aman_TxtBox.Text = "Actual Execution Time of Worst StreamSet = " + MaxExeSetDuration.ToString() + " || Optimum ExeSet includes: P" + MaxExeSetOrder[0].ToString() + "_P" + MaxExeSetOrder[1].ToString() + "_P" + MaxExeSetOrder[2].ToString() + " | P" + MaxExeSetOrder[3].ToString() + "_P" + MaxExeSetOrder[4].ToString() + "_P" + MaxExeSetOrder[5].ToString() + " | P" + MaxExeSetOrder[6].ToString() + "_P" + MaxExeSetOrder[7].ToString() + "_P" + MaxExeSetOrder[8].ToString();
            }
            ActualExeSet_Btn.Enabled = false;
        }

        private static int NumofStreamSets()
        {
            int StreamSets = CoScheduler.InputPrograms;
            int nextP = StreamSets;
            int Streams = CoScheduler.NumofStreams;
            for (int i = Streams; i > 1; i--)
            {
                nextP--;
                StreamSets *= nextP;
            }
            return StreamSets;
        }

        private void ExeSetGenerator(List<int[]> ExeSet)
        {
            //int tmp;
            //for (int i = 0; i < CoScheduler.InputPrograms / 3; i++)
            //{
            //    for (int j = 1; j <= CoScheduler.InputPrograms - (CoScheduler.InputPrograms / 3); j++)
            //    {
            //        for (int k = 1; k <= CoScheduler.InputPrograms - (CoScheduler.InputPrograms / 3); k++)
            //        {
            //            for (int z = 1; z <= CoScheduler.InputPrograms - (CoScheduler.InputPrograms / 3); z++)
            //            {
            //                if ((z != j) && (z != k) && (k != j))
            //                {

            //                }
            //            }
            //        }
            //    }
            //}
            //int cnt = 0;
            //for (int i = 1; i <= CoScheduler.InputPrograms; i++)
            //{
            //    for (int j = 1; j <= ExeSet.Count; j++)
            //    {
            //        if (ExeSet[i - 1] != ExeSet[j - 1])
            //        {
            //            cnt++;
            //        }
            //    }
            //    if (cnt == Num)
            //    {
            //        ExeSet.Add(i);
            //        ExeSetGenerator(ExeSet, i);
            //    }
            //}
            LinqDataContext DB = new LinqDataContext();
            //DB.ExecuteCommand("TRUNCATE TABLE PredictedExeSetTable");
            int[] streamsets = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            for (int a = 1; a <= CoScheduler.InputPrograms; a++)
            {
                for (int b = 1; b <= CoScheduler.InputPrograms; b++)
                {
                    if (a != b)
                    {
                        for (int c = 1; c <= CoScheduler.InputPrograms; c++)
                        {
                            if ((c != a) && (c != b))
                            {
                                for (int d = 1; d <= CoScheduler.InputPrograms; d++)
                                {
                                    if ((d != a) && (d != b) && (d != c))
                                    {
                                        for (int e = 1; e <= CoScheduler.InputPrograms; e++)
                                        {
                                            if ((e != a) && (e != b) && (e != c) && (e != d))
                                            {
                                                for (int f = 1; f <= CoScheduler.InputPrograms; f++)
                                                {
                                                    if ((f != a) && (f != b) && (f != c) && (f != d) && (f != e))
                                                    {
                                                        DB.SavePredictedExeSetTable(a, b, c, d, e, f, 0, 0, 0, 0, 0, 0, 0, 0);
                                                        streamsets[0] = a;
                                                        streamsets[1] = b;
                                                        streamsets[2] = c;
                                                        streamsets[3] = d;
                                                        streamsets[4] = e;
                                                        streamsets[5] = f;
                                                        ExeSet.Add(streamsets);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            //byte[] streamsets = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            //Parallel.For(1, 15, a =>
            //{
            //    for (byte b = 1; b <= CoScheduler.InputPrograms; b++)
            //    {
            //        if (a != b)
            //        {
            //            for (byte c = 1; c <= CoScheduler.InputPrograms; c++)
            //            {
            //                if ((c != a) && (c != b))
            //                {
            //                    for (byte d = 1; d <= CoScheduler.InputPrograms; d++)
            //                    {
            //                        if ((d != a) && (d != b) && (d != c))
            //                        {
            //                            for (byte e = 1; e <= CoScheduler.InputPrograms; e++)
            //                            {
            //                                if ((e != a) && (e != b) && (e != c) && (e != d))
            //                                {
            //                                    for (byte f = 1; f <= CoScheduler.InputPrograms; f++)
            //                                    {
            //                                        if ((f != a) && (f != b) && (f != c) && (f != d) && (f != e))
            //                                        {
            //                                            for (byte g = 1; g <= CoScheduler.InputPrograms; g++)
            //                                            {
            //                                                if ((g != a) && (g != b) && (g != c) && (g != d) && (g != e) && (g != f))
            //                                                {
            //                                                    for (byte h = 1; h <= CoScheduler.InputPrograms; h++)
            //                                                    {
            //                                                        if ((h != a) && (h != b) && (h != c) && (h != d) && (h != e) && (h != f) && (h != g))
            //                                                        {
            //                                                            for (byte i = 1; i <= CoScheduler.InputPrograms; i++)
            //                                                            {
            //                                                                if ((i != a) && (i != b) && (i != c) && (i != d) && (i != e) && (i != f) && (i != g) && (i != h))
            //                                                                {
            //                                                                    for (byte j = 1; j <= CoScheduler.InputPrograms; j++)
            //                                                                    {
            //                                                                        if ((j != a) && (j != b) && (j != c) && (j != d) && (j != e) && (j != f) && (j != g) && (j != h) && (j != i))
            //                                                                        {
            //                                                                            for (byte k = 1; k <= CoScheduler.InputPrograms; k++)
            //                                                                            {
            //                                                                                if ((k != a) && (k != b) && (k != c) && (k != d) && (k != e) && (k != f) && (k != g) && (k != h) && (k != i) && (k != j))
            //                                                                                {
            //                                                                                    for (byte l = 1; l <= CoScheduler.InputPrograms; l++)
            //                                                                                    {
            //                                                                                        if ((l != a) && (l != b) && (l != c) && (l != d) && (l != e) && (l != f) && (l != g) && (l != h) && (l != i) && (l != j) && (l != k))
            //                                                                                        {
            //                                                                                            for (byte m = 1; m <= CoScheduler.InputPrograms; m++)
            //                                                                                            {
            //                                                                                                if ((m != a) && (m != b) && (m != c) && (m != d) && (m != e) && (m != f) && (m != g) && (m != h) && (m != i) && (m != j) && (m != k) && (m != l))
            //                                                                                                {
            //                                                                                                    for (byte n = 1; n <= CoScheduler.InputPrograms; n++)
            //                                                                                                    {
            //                                                                                                        if ((n != a) && (n != b) && (n != c) && (n != d) && (n != e) && (n != f) && (n != g) && (n != h) && (n != i) && (n != j) && (n != k) && (n != l) && (n != m))
            //                                                                                                        {
            //                                                                                                            DB.SavePredictedExeSetTable(Convert.ToByte(a), b, c, d, e, f, g, h, i, j, k, l, m, n);
            //                                                                                                            //DB.SaveExeSetTable("S1= " + a.ToString() + " | S2= " + b.ToString() + " | S3= " + c.ToString(),
            //                                                                                                            //                   "S1= " + d.ToString() + " | S2= " + e.ToString() + " | S3= " + f.ToString(),
            //                                                                                                            //                   "S1= " + g.ToString() + " | S2= " + h.ToString() + " | S3= " + i.ToString(),
            //                                                                                                            //                   "S1= " + j.ToString() + " | S2= " + k.ToString() + " | S3= " + l.ToString(),
            //                                                                                                            //                   "S1= " + m.ToString() + " | S2= " + n.ToString());
            //                                                                                                            //streamsets[0] = a;
            //                                                                                                            //streamsets[1] = b;
            //                                                                                                            //streamsets[2] = c;
            //                                                                                                            //streamsets[3] = d;
            //                                                                                                            //streamsets[4] = e;
            //                                                                                                            //streamsets[5] = f;
            //                                                                                                            //streamsets[6] = g;
            //                                                                                                            //streamsets[7] = h;
            //                                                                                                            //streamsets[8] = i;
            //                                                                                                            //streamsets[9] = j;
            //                                                                                                            //streamsets[10] = k;
            //                                                                                                            //streamsets[11] = l;
            //                                                                                                            //streamsets[12] = m;
            //                                                                                                            //streamsets[13] = n;
            //                                                                                                            //ExeSet.Add(streamsets);
            //                                                                                                        }
            //                                                                                                    }
            //                                                                                                }
            //                                                                                            }
            //                                                                                        }
            //                                                                                    }
            //                                                                                }
            //                                                                            }
            //                                                                        }
            //                                                                    }
            //                                                                }
            //                                                            }
            //                                                        }
            //                                                    }
            //                                                }
            //                                            }
            //                                        }
            //                                    }
            //                                }
            //                            }
            //                        }
            //                    }
            //                }
            //            }
            //        }
            //    }
            //});
        }

        private void SelectActual4StreamSetFile_Btn_Click(object sender, EventArgs e)
        {
            ExeSetopenFileDialog.Filter = "Excel 97-2003 |*.xls|" + "Excel Workbook |*.xlsx";
            ExeSetopenFileDialog.FilterIndex = 2;
            ExeSetopenFileDialog.FileName = "";
            DialogResult DR = ExeSetopenFileDialog.ShowDialog();
            if (DR == DialogResult.OK)
            {
                String Path = ExeSetopenFileDialog.FileName;
                Actual2FileLocation_TxtBox.Text = Path;
                ExeSetConStr = @"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=" + Path + ";Extended Properties=Excel 12.0";
                ExeSetOleDbCon = new OleDbConnection(ExeSetConStr);
                ExeSetOleDbCon.Open();
                ExeSetUpdate_ComboBox(ActualFileWorksheet2_ComboBox);
                ExeSetOleDbCon.Close();
            }
            ImportActual2File_Btn.Enabled = false;
            ActualExeSet_Btn.Enabled = false;
        }

        private void Actual2FileLocation_TxtBox_TextChanged(object sender, EventArgs e)
        {

        }

        private void Actual4FileLocation_TxtBox_TextChanged(object sender, EventArgs e)
        {

        }

        private void Actual2FileWorksheet_Lbl_Click(object sender, EventArgs e)
        {

        }

        private void ActualFileWorksheet4_ComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            ImportActual4File_Btn.Enabled = true;
            ActualExeSet_Btn.Enabled = false;
        }

        private void ImportActual4File_Btn_Click(object sender, EventArgs e)
        {
            ExeSetOleDbCon.Open();
            OleDbCommand OleDbcmd = new OleDbCommand("Select * From [" + ActualFileWorksheet2_ComboBox.GetItemText(ActualFileWorksheet2_ComboBox.SelectedItem) + "$]", ExeSetOleDbCon);
            System.Data.DataTable DT = new System.Data.DataTable();
            DT.Load(OleDbcmd.ExecuteReader());
            Actual2StreamSetFile_GridView.DataSource = DT;
            ExeSetOleDbCon.Close();
            foreach (DataGridViewRow row in Actual2StreamSetFile_GridView.Rows)
            {
                try
                {
                    //AES_GridView.Rows[row.Index].Cells[CoScheduler.NumofStreams].Value = Math.Round((Convert.ToDouble(nonStream_GridView.Rows[row.Index].Cells[1].Value) - Convert.ToDouble(row.Cells[1].Value)), 2);
                    AES_GridView.Rows[row.Index].Cells[CoScheduler.NumofStreams].Value = Math.Round(Convert.ToDouble(row.Cells[1].Value), 2);
                }
                catch { }
            }
            ActualExeSet_Btn.Enabled = true;
            ImportActual2File_Btn.Enabled = false;
        }
    }
}
