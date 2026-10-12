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
    public partial class ActualExecutions : Form
    {
        public static String AEConStr;
        public static OleDbConnection AEOleDbCon;
        public static int S1 = 0, S2 = 0, S3 = 0;
        public ActualExecutions()
        {
            InitializeComponent();
        }

        private void ActualExecutions_Load(object sender, EventArgs e)
        {
            AEFileLoad_Btn.Enabled = false;
            AESort_Btn.Enabled = false;
            ExportSortedAE_Btn.Enabled = false;
        }

        private void AEFile_Btn_Click(object sender, EventArgs e)
        {
            AEopenFileDialog.Filter = "Excel 97-2003 |*.xls|" + "Excel Workbook |*.xlsx";
            AEopenFileDialog.FilterIndex = 2;
            AEopenFileDialog.FileName = "";
            DialogResult DR = AEopenFileDialog.ShowDialog();
            if (DR == DialogResult.OK)
            {
                String Path = AEopenFileDialog.FileName;
                AEFile_TxtBox.Text = Path;
                AEConStr = @"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=" + Path + ";Extended Properties=Excel 12.0";
                AEOleDbCon = new OleDbConnection(AEConStr);
                AEOleDbCon.Open();
                Update_ComboBox();
                AEOleDbCon.Close();
            }
        }

        private void AEFileLoad_Btn_Click(object sender, EventArgs e)
        {
            AEOleDbCon.Open();
            OleDbCommand OleDbcmd = new OleDbCommand("Select * From [" + AEWorkSheet_ComboBox.GetItemText(AEWorkSheet_ComboBox.SelectedItem) + "$]", AEOleDbCon);
            System.Data.DataTable DT = new System.Data.DataTable();
            DT.Load(OleDbcmd.ExecuteReader());
            AEFile_GridView.DataSource = DT;
            AEOleDbCon.Close();
            try
            {
                while (AEFile_GridView.Columns[4].Name.ToString() != string.Empty)
                {
                    AEFile_GridView.Columns.RemoveAt(4);
                }
            }
            catch
            {

            }
            AESort_Btn.Enabled = true;
            AEFileLoad_Btn.Enabled = false;
            ExportSortedAE_Btn.Enabled = false;
        }

        private void Update_ComboBox()
        {
            AEWorkSheet_ComboBox.Items.Clear();
            AEWorkSheet_ComboBox.ResetText();
            System.Data.DataTable Sheet_DT = new System.Data.DataTable();
            Sheet_DT = AEOleDbCon.GetOleDbSchemaTable(OleDbSchemaGuid.Tables, null);
            foreach (DataRow row in Sheet_DT.Rows)
            {
                AEWorkSheet_ComboBox.Items.Add(row["TABLE_NAME"].ToString().Replace("$", string.Empty));
            }
        }

        private void AEWorkSheet_ComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            AEFileLoad_Btn.Enabled = true;
            ExportSortedAE_Btn.Enabled = false;
        }

        private void AESort_Btn_Click(object sender, EventArgs e)
        {
            int StreamSetNum = 0;
            Int64 PipelineStart = Convert.ToInt64(AEFile_GridView.Rows[0].Cells["Start Time(ns)"].Value), PipelineStop = 0;
            for (int i = 1; i < AEFile_GridView.Rows.Count; i++)
            {
                if (Convert.ToDouble(AEFile_GridView.Rows[i].Cells["Difference(ns)"].Value) > 0.8)
                {
                    PipelineStop = Convert.ToInt64(AEFile_GridView.Rows[i].Cells["Start Time(ns)"].Value) + Convert.ToInt64(AEFile_GridView.Rows[i].Cells["Duration(ns)"].Value);
                    StreamSet_detector(StreamSetNum + 1);
                    AE_GridView.Rows.Add();
                    AE_GridView.Rows[StreamSetNum].Cells[0].Value = "S1= " + S1.ToString() + " | S2= " + S2.ToString() + " | S3= " + S3.ToString();
                    AE_GridView.Rows[StreamSetNum].Cells[1].Value = Convert.ToDouble(PipelineStop - PipelineStart) / 1000000;
                    if (i + 1 < AEFile_GridView.Rows.Count)
                    {
                        PipelineStart = Convert.ToInt64(AEFile_GridView.Rows[i + 1].Cells["Start Time(ns)"].Value);
                    }
                    StreamSetNum++;
                }
            }
            ExportSortedAE_Btn.Enabled = true;
        }

        private void ExportSortedAE_Btn_Click(object sender, EventArgs e)
        {
            Microsoft.Office.Interop.Excel.Application AEApp = new Microsoft.Office.Interop.Excel.Application();
            _Workbook AEWorkbook = AEApp.Workbooks.Add(Type.Missing);
            _Worksheet AEWorksheet = null;
            AEWorksheet = AEWorkbook.Sheets["Sheet1"];
            AEWorksheet = AEWorkbook.ActiveSheet;
            AEWorksheet.Name = "ActualStreamSets";
            for (int i = 1; i < AEFile_GridView.Columns.Count + 1; i++)
            {
                try
                {
                    AEWorksheet.Cells[i, 1] = AE_GridView.Columns[i - 1].HeaderText;
                }
                catch { }
            }
            for(int i=0;i<AE_GridView.Rows.Count;i++)
            {
                for (int j = 0; j < AE_GridView.Columns.Count; j++)
                {
                    AEWorksheet.Cells[i + 2, j + 1] = AE_GridView.Rows[i].Cells[j].Value.ToString();
                }
            }
            AEsaveFileDialog.FileName = "ExportedActualExecutions";
            AEsaveFileDialog.DefaultExt = ".xlsx";
            if (AEsaveFileDialog.ShowDialog() == DialogResult.OK)
            {
                AEWorkbook.SaveAs(AEsaveFileDialog.FileName, Type.Missing, Type.Missing, Type.Missing, Type.Missing, Type.Missing, XlSaveAsAccessMode.xlExclusive, Type.Missing, Type.Missing, Type.Missing, Type.Missing, Type.Missing);
            }
            AEApp.Quit();
        }

        private void StreamSet_detector(int row)
        {
            int cntr = 0;
            for (int i = 1; i <= CoScheduler.InputPrograms; i++)
            {
                for (int j = 1; j <= CoScheduler.InputPrograms; j++)
                {
                    for (int k = 1; k <= CoScheduler.InputPrograms; k++)
                    {
                        if (i != j && i != k && j != k)
                        {
                            cntr++;
                            if (cntr == row)
                            {
                                S1 = i;
                                S2 = j;
                                S3 = k;
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}
