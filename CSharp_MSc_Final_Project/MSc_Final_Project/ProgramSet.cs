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
    public partial class ProgramSet : Form
    {
        public static String PSConStr;
        public static OleDbConnection PSOleDbCon;
        public ProgramSet()
        {
            InitializeComponent();
        }

        private void ProgramSet_Load(object sender, EventArgs e)
        {
            PSFileLoad_Btn.Enabled = false;
            SavePS_Btn.Enabled = false;
        }

        private void PSFile_Btn_Click(object sender, EventArgs e)
        {
            SavePS_Btn.Enabled = false;
            PSopenFileDialog.Filter = "Excel 97-2003 |*.xls|" + "Excel Workbook |*.xlsx";
            PSopenFileDialog.FilterIndex = 2;
            PSopenFileDialog.FileName = "";
            DialogResult DR = PSopenFileDialog.ShowDialog();
            if (DR == DialogResult.OK)
            {
                String Path = PSopenFileDialog.FileName;
                PSFileLocation_TxtBox.Text = Path;
                PSConStr = @"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=" + Path + ";Extended Properties=Excel 12.0";
                PSOleDbCon = new OleDbConnection(PSConStr);
                PSOleDbCon.Open();
                PSUpdate_ComboBox();
                PSOleDbCon.Close();
            }
        }

        private void PSWorksheet_ComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            PSFileLoad_Btn.Enabled = true;
            SavePS_Btn.Enabled = false;
        }

        private void PSFileLoad_Btn_Click(object sender, EventArgs e)
        {
            PSOleDbCon.Open();
            OleDbCommand OleDbcmd = new OleDbCommand("Select * From [" + PSWorksheet_ComboBox.GetItemText(PSWorksheet_ComboBox.SelectedItem) + "$]", PSOleDbCon);
            System.Data.DataTable DT = new System.Data.DataTable();
            DT.Load(OleDbcmd.ExecuteReader());
            PS_GridView.DataSource = DT;
            PSOleDbCon.Close();
            foreach (DataGridViewRow row in PS_GridView.Rows)
            {
                for (int i = 0; i <= 16; i++)
                {
                    if (row.Cells[i].Value.ToString() == string.Empty)
                    {
                        row.Cells[i].Value = 0;
                    }
                }
            }
            PSFileLoad_Btn.Enabled = false;
            SavePS_Btn.Enabled = true;
        }
        private void PSUpdate_ComboBox()
        {
            PSWorksheet_ComboBox.Items.Clear();
            PSWorksheet_ComboBox.ResetText();
            System.Data.DataTable Sheet_DT = new System.Data.DataTable();
            Sheet_DT = PSOleDbCon.GetOleDbSchemaTable(OleDbSchemaGuid.Tables, null);
            foreach (DataRow row in Sheet_DT.Rows)
            {
                PSWorksheet_ComboBox.Items.Add(row["TABLE_NAME"].ToString().Replace("$", string.Empty));
            }
        }
        private void SavePS_Btn_Click(object sender, EventArgs e)
        {
            SavePS_Btn.Enabled = false;
            double MemcpyH2D, Kernel, MemcpyD2H, PipelineDuration_nonStream;
            LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
            DB.ExecuteCommand("TRUNCATE TABLE IIIStreamSetTable");
            DB.ExecuteCommand("TRUNCATE TABLE IIStreamSetTable");
            for (int i = 1; i <= 100; i++)
            {
                try
                {
                    DB.DeletePSTable(i);
                }
                catch { }
            }
            foreach (DataGridViewRow row in PS_GridView.Rows)
            {
                try
                {
                    PipelineDuration_nonStream = (((Convert.ToInt32(row.Cells["MemcpyH2D_1"].Value) +
                                                    Convert.ToInt32(row.Cells["MemcpyH2D_2"].Value) +
                                                    Convert.ToInt32(row.Cells["MemcpyH2D_3"].Value) +
                                                    Convert.ToInt32(row.Cells["MemcpyH2D_4"].Value)) / CoScheduler.H2DSynccpyBW) * 1000) +
                                                    (((Convert.ToDouble(row.Cells["Kernel_1_Duration"].Value) * Convert.ToDouble(row.Cells["Kernel_1_Count"].Value)) +
                                                    (Convert.ToDouble(row.Cells["Kernel_2_Duration"].Value) * Convert.ToDouble(row.Cells["Kernel_2_Count"].Value)) +
                                                    (Convert.ToDouble(row.Cells["Kernel_3_Duration"].Value) * Convert.ToDouble(row.Cells["Kernel_3_Count"].Value)) +
                                                    (Convert.ToDouble(row.Cells["Kernel_4_Duration"].Value) * Convert.ToDouble(row.Cells["Kernel_4_Count"].Value))) / 1000000) +
                                                    (((Convert.ToInt32(row.Cells["MemcpyD2H_1"].Value) +
                                                    Convert.ToInt32(row.Cells["MemcpyD2H_2"].Value) +
                                                    Convert.ToInt32(row.Cells["MemcpyD2H_3"].Value) +
                                                    Convert.ToInt32(row.Cells["MemcpyD2H_4"].Value)) / CoScheduler.D2HSynccpyBW) * 1000);
                    DB.SavenonStreamTable((row.Index + 1), PipelineDuration_nonStream);
                }
                catch { }
                try
                {
                    MemcpyH2D = Math.Abs(((Convert.ToDouble(row.Cells["MemcpyH2D_1"].Value) +
                                   Convert.ToDouble(row.Cells["MemcpyH2D_2"].Value) +
                                   Convert.ToDouble(row.Cells["MemcpyH2D_3"].Value) +
                                   Convert.ToDouble(row.Cells["MemcpyH2D_4"].Value)) / Convert.ToDouble(CoScheduler.H2DAsynccpyBW)) * 1000);
                    Kernel = (((Convert.ToDouble(row.Cells["Kernel_1_Duration"].Value) * Convert.ToDouble(row.Cells["Kernel_1_Count"].Value)) +
                                   (Convert.ToDouble(row.Cells["Kernel_2_Duration"].Value) * Convert.ToDouble(row.Cells["Kernel_2_Count"].Value)) +
                                   (Convert.ToDouble(row.Cells["Kernel_3_Duration"].Value) * Convert.ToDouble(row.Cells["Kernel_3_Count"].Value)) +
                                   (Convert.ToDouble(row.Cells["Kernel_4_Duration"].Value) * Convert.ToDouble(row.Cells["Kernel_4_Count"].Value))) / 1000000);
                    MemcpyD2H = Math.Abs(((Convert.ToDouble(row.Cells["MemcpyD2H_1"].Value) +
                                   Convert.ToDouble(row.Cells["MemcpyD2H_2"].Value) +
                                   Convert.ToDouble(row.Cells["MemcpyD2H_3"].Value) +
                                   Convert.ToDouble(row.Cells["MemcpyD2H_4"].Value)) / Convert.ToDouble(CoScheduler.D2HAsynccpyBW)) * 1000);
                    DB.SavePSTable((row.Index + 1), MemcpyH2D, Kernel, MemcpyD2H);
                }
                catch { }
            }
            MessageBox.Show("Successfully Saved!", "Information", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }
}
