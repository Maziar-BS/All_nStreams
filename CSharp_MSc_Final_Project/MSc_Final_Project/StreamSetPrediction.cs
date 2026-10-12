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
    public partial class StreamSetPrediction : Form
    {
        public static String ASConStr;
        public static OleDbConnection ASOleDbCon;
        public static bool ErrBtnCondition_1, ErrBtnCondition_2;
        public StreamSetPrediction()
        {
            InitializeComponent();
        }

        private void StreamSetPrediction_Load(object sender, EventArgs e)
        {
            ExportnonStreamBtn.Enabled = false;
            LoadASFile_Btn.Enabled = false;
            //PredictionResults_Btn.Enabled = false;
            ErrBtnCondition_1 = false;
            ErrBtnCondition_2 = false;
            StreamSetsError_Btn.Enabled = false;
        }

        private void nonStreamBtn_Click(object sender, EventArgs e)
        {
            LinqDataContext DB = new LinqDataContext();
            var P2 = DB.PSTables.Where(P => P.ID == 1).Select(P => P).Single();
            double nonStreamPipeline;
            int row = 0;
            for (int i = 1; i <= CoScheduler.InputPrograms; i++)
            {
                for (int j = 1; j <= CoScheduler.InputPrograms; j++)
                {
                    for (int k = 1; k <= CoScheduler.InputPrograms; k++)
                    {
                        if (i != j && i != k && j != k)
                        {
                            nonStream_GridView.Rows.Add();
                            nonStream_GridView.Rows[row].Cells[0].Value = "S1= " + i.ToString() +
                                                                            " | S2= " + j.ToString() +
                                                                            " | S3= " + k.ToString();
                            nonStreamPipeline = DB.nonStreamTables.Where(P => P.ID == i).Select(P => P.PipelineDuration).Single() +
                                                DB.nonStreamTables.Where(P => P.ID == j).Select(P => P.PipelineDuration).Single() +
                                                DB.nonStreamTables.Where(P => P.ID == k).Select(P => P.PipelineDuration).Single();
                            nonStream_GridView.Rows[row].Cells[1].Value = nonStreamPipeline;
                            row++;
                        }
                    }
                }
            }
            ExportnonStreamBtn.Enabled = true;
        }

        private void ExportnonStreamBtn_Click(object sender, EventArgs e)
        {
            Microsoft.Office.Interop.Excel.Application nSSApp = new Microsoft.Office.Interop.Excel.Application();
            _Workbook nSSWorkbook = nSSApp.Workbooks.Add(Type.Missing);
            _Worksheet nSSWorksheet = null;
            nSSWorksheet = nSSWorkbook.Sheets["Sheet1"];
            nSSWorksheet = nSSWorkbook.ActiveSheet;
            nSSWorksheet.Name = "nonStreamSets";
            for (int i = 1; i < nonStream_GridView.Columns.Count + 1; i++)
            {
                try
                {
                    nSSWorksheet.Cells[i, 1] = nonStream_GridView.Columns[i - 1].HeaderText;
                }
                catch { }
            }
            for (int i = 0; i < nonStream_GridView.Rows.Count; i++)
            {
                for (int j = 0; j < nonStream_GridView.Columns.Count; j++)
                {
                    nSSWorksheet.Cells[i + 2, j + 1] = nonStream_GridView.Rows[i].Cells[j].Value.ToString();
                }
            }
            StreamSetsaveFileDialog.FileName = "ExportednonStreamSets";
            StreamSetsaveFileDialog.DefaultExt = ".xlsx";
            if (StreamSetsaveFileDialog.ShowDialog() == DialogResult.OK)
            {
                nSSWorkbook.SaveAs(StreamSetsaveFileDialog.FileName, Type.Missing, Type.Missing, Type.Missing, Type.Missing, Type.Missing, XlSaveAsAccessMode.xlExclusive, Type.Missing, Type.Missing, Type.Missing, Type.Missing, Type.Missing);
            }
            nSSApp.Quit();
            ExportnonStreamBtn.Enabled = false;
        }

        private void PredictStreamSets_Btn_Click(object sender, EventArgs e)
        {
            IIStreamSetGenerator();
            PredictionProgress_Lbl.Text = "Generating 3xStream Sets..";
            IIIStreamSetGenerator();
            //IVStreamSetGenerator();
            //VStreamSetGenerator();
            //VIStreamSetGenerator();
            //VIIStreamSetGenerator();
            //VIIIStreamSetGenerator();
            //IXStreamSetGenerator();
            //XStreamSetGenerator();
            //XIStreamSetGenerator();
            //XIIStreamSetGenerator();
            //XIIIStreamSetGenerator();
            //XIVStreamSetGenerator();
            PredictionProgress_Lbl.Text = "Finished!";
            PredictionResults_Btn.Enabled = true;
        }

        private void PredictionResults_Btn_Click(object sender, EventArgs e)
        {
            switch (CoScheduler.NumofCombs)
            {
                case 1:
                    {
                        break;
                    }
                case 2:
                    {
                        break;
                    }
                case 3:
                    {
                        StreamSet_detector();
                        break;
                    }
            }
            ErrBtnCondition_1 = true;
            if (ErrBtnCondition_1 && ErrBtnCondition_2)
            {
                StreamSetsError_Btn.Enabled = true;
            }
            else
            {
                StreamSetsError_Btn.Enabled = false;
            }
            //PredictionResults_Btn.Enabled = false;
        }

        private void SelectASFile_Btn_Click(object sender, EventArgs e)
        {
            LoadASFile_Btn.Enabled = false;
            ErrBtnCondition_2 = false;
            ASopenFileDialog.Filter = "Excel 97-2003 |*.xls|" + "Excel Workbook |*.xlsx";
            ASopenFileDialog.FilterIndex = 2;
            ASopenFileDialog.FileName = "";
            DialogResult DR = ASopenFileDialog.ShowDialog();
            if (DR == DialogResult.OK)
            {
                String Path = ASopenFileDialog.FileName;
                ASFileLocation_TxtBot.Text = Path;
                ASConStr = @"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=" + Path + ";Extended Properties=Excel 12.0";
                ASOleDbCon = new OleDbConnection(ASConStr);
                ASOleDbCon.Open();
                ASUpdate_ComboBox();
                ASOleDbCon.Close();
            }
        }

        private void ASUpdate_ComboBox()
        {
            ASFileWorksheet_ComboBox.Items.Clear();
            ASFileWorksheet_ComboBox.ResetText();
            System.Data.DataTable Sheet_DT = new System.Data.DataTable();
            Sheet_DT = ASOleDbCon.GetOleDbSchemaTable(OleDbSchemaGuid.Tables, null);
            foreach (DataRow row in Sheet_DT.Rows)
            {
                ASFileWorksheet_ComboBox.Items.Add(row["TABLE_NAME"].ToString().Replace("$", string.Empty));
            }
        }

        private void ASFileWorksheet_ComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            LoadASFile_Btn.Enabled = true;
            ErrBtnCondition_2 = false;
        }

        private void LoadASFile_Btn_Click(object sender, EventArgs e)
        {
            ASOleDbCon.Open();
            OleDbCommand OleDbcmd = new OleDbCommand("Select * From [" + ASFileWorksheet_ComboBox.GetItemText(ASFileWorksheet_ComboBox.SelectedItem) + "$]", ASOleDbCon);
            System.Data.DataTable DT = new System.Data.DataTable();
            DT.Load(OleDbcmd.ExecuteReader());
            ASFile_GridView.DataSource = DT;
            ASOleDbCon.Close();
            LoadASFile_Btn.Enabled = false;
            ErrBtnCondition_2 = true;
            if (ErrBtnCondition_1 && ErrBtnCondition_2)
            {
                StreamSetsError_Btn.Enabled = true;
            }
            else
            {
                StreamSetsError_Btn.Enabled = false;
            }
        }

        private void StreamSetsError_Btn_Click(object sender, EventArgs e)
        {
            double ActualValue, PredictedValue;
            for (int i = 0; i < NumofStreamSets(CoScheduler.InputPrograms, CoScheduler.NumofCombs); i++)
            {
                StreamSetsError_GridView.Rows.Add();
                ActualValue = Math.Round(Convert.ToDouble(ASFile_GridView.Rows[i].Cells[1].Value), 3);
                PredictedValue = Math.Round(Convert.ToDouble(PredictedStreamSets_GridView.Rows[i].Cells[1].Value), 3);
                StreamSetsError_GridView.Rows[i].Cells[0].Value = Math.Abs(((ActualValue - PredictedValue) / ActualValue) * 100);
            }
            ErrBtnCondition_1 = false;
            ErrBtnCondition_2 = false;
            StreamSetsError_Btn.Enabled = false;
        }

        private void ExportStreamSet_Btn_Click(object sender, EventArgs e)
        {
            Microsoft.Office.Interop.Excel.Application PSSApp = new Microsoft.Office.Interop.Excel.Application();
            _Workbook PSSWorkbook = PSSApp.Workbooks.Add(Type.Missing);
            _Worksheet PSSWorksheet = null;
            PSSWorksheet = PSSWorkbook.Sheets["Sheet1"];
            PSSWorksheet = PSSWorkbook.ActiveSheet;
            PSSWorksheet.Name = "PredictedStreamSets";
            for (int i = 1; i < PredictedStreamSets_GridView.Columns.Count + 1; i++)
            {
                try
                {
                    PSSWorksheet.Cells[i, 1] = PredictedStreamSets_GridView.Columns[i - 1].HeaderText;
                }
                catch { }
            }
            for (int i = 0; i < PredictedStreamSets_GridView.Rows.Count; i++)
            {
                for (int j = 0; j < PredictedStreamSets_GridView.Columns.Count; j++)
                {
                    PSSWorksheet.Cells[i + 2, j + 1] = PredictedStreamSets_GridView.Rows[i].Cells[j].Value.ToString();
                }
            }
            StreamSetsaveFileDialog.FileName = "ExportedStreamSetsPrediction";
            StreamSetsaveFileDialog.DefaultExt = ".xlsx";
            if (StreamSetsaveFileDialog.ShowDialog() == DialogResult.OK)
            {
                PSSWorkbook.SaveAs(StreamSetsaveFileDialog.FileName, Type.Missing, Type.Missing, Type.Missing, Type.Missing, Type.Missing, XlSaveAsAccessMode.xlExclusive, Type.Missing, Type.Missing, Type.Missing, Type.Missing, Type.Missing);
            }
            PSSApp.Quit();
        }

        private void StreamSet_detector()
        {
            LinqDataContext DB = new LinqDataContext();
            IIStreamSetTable IIStreamSetDetector = new IIStreamSetTable();
            IIIStreamSetTable IIIStreamSetDetector = new IIIStreamSetTable();
            var P2 = DB.PSTables.Where(P => P.ID == 1).Select(P => P).Single();
            int PID1, PID2, PID3;
            int row = 0;
            for (int i = 1; i <= CoScheduler.InputPrograms; i++)
            {
                for (int j = 1; j <= CoScheduler.InputPrograms; j++)
                {
                    for (int k = 1; k <= CoScheduler.InputPrograms; k++)
                    {
                        if (i != j && i != k && j != k)
                        {
                            PredictedStreamSets_GridView.Rows.Add();
                            PredictedStreamSets_GridView.Rows[row].Cells[0].Value = "S1= " + i.ToString() +
                                                                            " | S2= " + j.ToString() +
                                                                            " | S3= " + k.ToString();
                            PID1 = DB.PSTables.Where(P => P.ID == i).Select(P => P.ID).Single();
                            PID2 = DB.PSTables.Where(P => P.ID == j).Select(P => P.ID).Single();
                            PID3 = DB.PSTables.Where(P => P.ID == k).Select(P => P.ID).Single();
                            IIStreamSetDetector = DB.IIStreamSetTables.Where(P => (P.PID1 == PID1) && (P.PID2 == PID2)).Select(P => P).Single();
                            IIIStreamSetDetector = DB.IIIStreamSetTables.Where(P => (P.PID1 == IIStreamSetDetector.ID) && (P.PID2 == PID3)).Select(P => P).Single();
                            PredictedStreamSets_GridView.Rows[row].Cells[1].Value = IIIStreamSetDetector.MemcpyH2D + IIIStreamSetDetector.Kernel + IIIStreamSetDetector.MemcpyD2H;
                            row++;
                        }
                    }
                }
            }
        }

        private static int NumofStreamSets(int Programs, int Streams)
        {
            int NumofPrograms = Programs;
            for (int i = 1; i < Streams; i++)
            {
                NumofPrograms--;
                Programs *= NumofPrograms;
            }
            return Programs;
        }

        private void IIStreamSetGenerator()
        {
            LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
            PredictionProgress_Lbl.Text = "Truncating all StreamSets records..";
            DB.ExecuteCommand("TRUNCATE TABLE IIIStreamSetTable");
            DB.ExecuteCommand("TRUNCATE TABLE IIStreamSetTable");
            //for (int i = 0; i < 3000; i++)
            //{
            //    DB.DeleteIIStreamTable(i);
            //}
            PredictionProgress_Lbl.Text = "Generating 2xStream Sets..";
            var P1 = DB.PSTables.Where(P => P.ID == 1).Select(P => P).Single();
            var P2 = DB.PSTables.Where(P => P.ID == 1).Select(P => P).Single();
            double H2D, Kernel, D2H;
            for (int i = 1; i <= CoScheduler.InputPrograms; i++)
            {
                for (int j = 1; j <= CoScheduler.InputPrograms; j++)
                {
                    if (i != j)
                    {
                        P1 = DB.PSTables.Where(P => P.ID == i).Select(P => P).Single();
                        P2 = DB.PSTables.Where(P => P.ID == j).Select(P => P).Single();
                        if (P1.Kernel >= P2.MemcpyH2D)
                        {
                            if (P1.MemcpyD2H >= P2.Kernel)
                            {
                                H2D = P1.MemcpyH2D + P2.MemcpyH2D;
                                Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
                                D2H = (P1.MemcpyD2H - P2.Kernel) + P2.MemcpyD2H;
                                DB.SaveIIStreamSetTable(H2D, Kernel, D2H, P1.ID, P2.ID);
                            }
                            else
                            {
                                H2D = P1.MemcpyH2D + P2.MemcpyH2D;
                                Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
                                D2H = P2.MemcpyD2H;
                                DB.SaveIIStreamSetTable(H2D, Kernel, D2H, P1.ID, P2.ID);
                            }
                        }
                        else
                        {
                            if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
                            {
                                H2D = P1.MemcpyH2D + P2.MemcpyH2D;
                                Kernel = P2.Kernel;
                                D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
                                DB.SaveIIStreamSetTable(H2D, Kernel, D2H, P1.ID, P2.ID);
                            }
                            else
                            {
                                H2D = P1.MemcpyH2D + P2.MemcpyH2D;
                                Kernel = P2.Kernel;
                                D2H = P2.MemcpyD2H;
                                DB.SaveIIStreamSetTable(H2D, Kernel, D2H, P1.ID, P2.ID);
                            }
                        }
                    }
                }
            }
        }

        private void IIIStreamSetGenerator()
        {
            LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
            IIStreamSetTable P1 = new IIStreamSetTable();
            var P2 = DB.PSTables.Where(P => P.ID == 1).Select(P => P).Single();
            double H2D, Kernel, D2H;
            //int PID1, PID2;
            for (int i = 1; i <= CoScheduler.InputPrograms; i++)
            {
                for (int j = 1; j <= CoScheduler.InputPrograms; j++)
                {
                    for (int k = 1; k <= CoScheduler.InputPrograms; k++)
                    {
                        if (i != j && i != k && j != k)
                        {
                            //PID1 = DB.PSTables.Where(P => P.ID == i).Select(P => P.ID).Single();
                            //PID2 = DB.PSTables.Where(P => P.ID == j).Select(P => P.ID).Single();
                            P1 = DB.IIStreamSetTables.Where(P => (P.PID1 == i) && (P.PID2 == j)).Select(P => P).Single();
                            P2 = DB.PSTables.Where(P => P.ID == k).Select(P => P).Single();
                            if (P1.Kernel >= P2.MemcpyH2D)
                            {
                                if (P1.MemcpyD2H >= P2.Kernel)
                                {
                                    H2D = P1.MemcpyH2D + P2.MemcpyH2D;
                                    Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
                                    D2H = (P1.MemcpyD2H - P2.Kernel) + P2.MemcpyD2H;
                                    DB.SaveIIIStreamSetTable(H2D, Kernel, D2H, P1.ID, P2.ID);
                                }
                                else
                                {
                                    H2D = P1.MemcpyH2D + P2.MemcpyH2D;
                                    Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
                                    D2H = P2.MemcpyD2H;
                                    DB.SaveIIIStreamSetTable(H2D, Kernel, D2H, P1.ID, P2.ID);
                                }
                            }
                            else
                            {
                                if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
                                {
                                    H2D = P1.MemcpyH2D + P2.MemcpyH2D;
                                    Kernel = P2.Kernel;
                                    D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
                                    DB.SaveIIIStreamSetTable(H2D, Kernel, D2H, P1.ID, P2.ID);
                                }
                                else
                                {
                                    H2D = P1.MemcpyH2D + P2.MemcpyH2D;
                                    Kernel = P2.Kernel;
                                    D2H = P2.MemcpyD2H;
                                    DB.SaveIIIStreamSetTable(H2D, Kernel, D2H, P1.ID, P2.ID);
                                }
                            }
                        }
                    }
                }
            }
        }

        //private void IVStreamSetGenerator()
        //{
        //    LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
        //    for (int i = 0; i < 1000; i++)
        //    {
        //        //DB.(i);
        //    }
        //    var P1 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    var P2 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    double H2D, Kernel, D2H;
        //    for (int i = 1; i <= CoScheduler.InputPrograms; i++)
        //    {
        //        for (int j = 1; j <= CoScheduler.InputPrograms; j++)
        //        {
        //            if (i != j)
        //            {
        //                P1 = DB./*PSTables*/.Where(P => P.ID == i).Select(P => P).Single();
        //                P2 = DB./*PSTables*/.Where(P => P.ID == j).Select(P => P).Single();
        //                if (P1.Kernel >= P2.MemcpyH2D)
        //                {
        //                    if (P1.MemcpyD2H >= P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //                else
        //                {
        //                    if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}

        //private void VStreamSetGenerator()
        //{
        //    LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
        //    for (int i = 0; i < 1000; i++)
        //    {
        //        //DB.(i);
        //    }
        //    var P1 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    var P2 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    double H2D, Kernel, D2H;
        //    for (int i = 1; i <= CoScheduler.InputPrograms; i++)
        //    {
        //        for (int j = 1; j <= CoScheduler.InputPrograms; j++)
        //        {
        //            if (i != j)
        //            {
        //                P1 = DB./*PSTables*/.Where(P => P.ID == i).Select(P => P).Single();
        //                P2 = DB./*PSTables*/.Where(P => P.ID == j).Select(P => P).Single();
        //                if (P1.Kernel >= P2.MemcpyH2D)
        //                {
        //                    if (P1.MemcpyD2H >= P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //                else
        //                {
        //                    if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}

        //private void VIStreamSetGenerator()
        //{
        //    LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
        //    for (int i = 0; i < 1000; i++)
        //    {
        //        //DB.(i);
        //    }
        //    var P1 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    var P2 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    double H2D, Kernel, D2H;
        //    for (int i = 1; i <= CoScheduler.InputPrograms; i++)
        //    {
        //        for (int j = 1; j <= CoScheduler.InputPrograms; j++)
        //        {
        //            if (i != j)
        //            {
        //                P1 = DB./*PSTables*/.Where(P => P.ID == i).Select(P => P).Single();
        //                P2 = DB./*PSTables*/.Where(P => P.ID == j).Select(P => P).Single();
        //                if (P1.Kernel >= P2.MemcpyH2D)
        //                {
        //                    if (P1.MemcpyD2H >= P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //                else
        //                {
        //                    if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}

        //private void VIIStreamSetGenerator()
        //{
        //    LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
        //    for (int i = 0; i < 1000; i++)
        //    {
        //        //DB.(i);
        //    }
        //    var P1 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    var P2 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    double H2D, Kernel, D2H;
        //    for (int i = 1; i <= CoScheduler.InputPrograms; i++)
        //    {
        //        for (int j = 1; j <= CoScheduler.InputPrograms; j++)
        //        {
        //            if (i != j)
        //            {
        //                P1 = DB./*PSTables*/.Where(P => P.ID == i).Select(P => P).Single();
        //                P2 = DB./*PSTables*/.Where(P => P.ID == j).Select(P => P).Single();
        //                if (P1.Kernel >= P2.MemcpyH2D)
        //                {
        //                    if (P1.MemcpyD2H >= P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //                else
        //                {
        //                    if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}

        //private void VIIIStreamSetGenerator()
        //{
        //    LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
        //    for (int i = 0; i < 1000; i++)
        //    {
        //        //DB.(i);
        //    }
        //    var P1 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    var P2 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    double H2D, Kernel, D2H;
        //    for (int i = 1; i <= CoScheduler.InputPrograms; i++)
        //    {
        //        for (int j = 1; j <= CoScheduler.InputPrograms; j++)
        //        {
        //            if (i != j)
        //            {
        //                P1 = DB./*PSTables*/.Where(P => P.ID == i).Select(P => P).Single();
        //                P2 = DB./*PSTables*/.Where(P => P.ID == j).Select(P => P).Single();
        //                if (P1.Kernel >= P2.MemcpyH2D)
        //                {
        //                    if (P1.MemcpyD2H >= P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //                else
        //                {
        //                    if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}

        //private void IXStreamSetGenerator()
        //{
        //    LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
        //    for (int i = 0; i < 1000; i++)
        //    {
        //        //DB.(i);
        //    }
        //    var P1 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    var P2 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    double H2D, Kernel, D2H;
        //    for (int i = 1; i <= CoScheduler.InputPrograms; i++)
        //    {
        //        for (int j = 1; j <= CoScheduler.InputPrograms; j++)
        //        {
        //            if (i != j)
        //            {
        //                P1 = DB./*PSTables*/.Where(P => P.ID == i).Select(P => P).Single();
        //                P2 = DB./*PSTables*/.Where(P => P.ID == j).Select(P => P).Single();
        //                if (P1.Kernel >= P2.MemcpyH2D)
        //                {
        //                    if (P1.MemcpyD2H >= P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //                else
        //                {
        //                    if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}

        //private void XStreamSetGenerator()
        //{
        //    LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
        //    for (int i = 0; i < 1000; i++)
        //    {
        //        //DB.(i);
        //    }
        //    var P1 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    var P2 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    double H2D, Kernel, D2H;
        //    for (int i = 1; i <= CoScheduler.InputPrograms; i++)
        //    {
        //        for (int j = 1; j <= CoScheduler.InputPrograms; j++)
        //        {
        //            if (i != j)
        //            {
        //                P1 = DB./*PSTables*/.Where(P => P.ID == i).Select(P => P).Single();
        //                P2 = DB./*PSTables*/.Where(P => P.ID == j).Select(P => P).Single();
        //                if (P1.Kernel >= P2.MemcpyH2D)
        //                {
        //                    if (P1.MemcpyD2H >= P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //                else
        //                {
        //                    if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}

        //private void XIStreamSetGenerator()
        //{
        //    LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
        //    for (int i = 0; i < 1000; i++)
        //    {
        //        //DB.(i);
        //    }
        //    var P1 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    var P2 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    double H2D, Kernel, D2H;
        //    for (int i = 1; i <= CoScheduler.InputPrograms; i++)
        //    {
        //        for (int j = 1; j <= CoScheduler.InputPrograms; j++)
        //        {
        //            if (i != j)
        //            {
        //                P1 = DB./*PSTables*/.Where(P => P.ID == i).Select(P => P).Single();
        //                P2 = DB./*PSTables*/.Where(P => P.ID == j).Select(P => P).Single();
        //                if (P1.Kernel >= P2.MemcpyH2D)
        //                {
        //                    if (P1.MemcpyD2H >= P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //                else
        //                {
        //                    if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}

        //private void XIIStreamSetGenerator()
        //{
        //    LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
        //    for (int i = 0; i < 1000; i++)
        //    {
        //        //DB.(i);
        //    }
        //    var P1 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    var P2 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    double H2D, Kernel, D2H;
        //    for (int i = 1; i <= CoScheduler.InputPrograms; i++)
        //    {
        //        for (int j = 1; j <= CoScheduler.InputPrograms; j++)
        //        {
        //            if (i != j)
        //            {
        //                P1 = DB./*PSTables*/.Where(P => P.ID == i).Select(P => P).Single();
        //                P2 = DB./*PSTables*/.Where(P => P.ID == j).Select(P => P).Single();
        //                if (P1.Kernel >= P2.MemcpyH2D)
        //                {
        //                    if (P1.MemcpyD2H >= P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //                else
        //                {
        //                    if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}

        //private void XIIIStreamSetGenerator()
        //{
        //    LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
        //    for (int i = 0; i < 1000; i++)
        //    {
        //        //DB.(i);
        //    }
        //    var P1 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    var P2 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    double H2D, Kernel, D2H;
        //    for (int i = 1; i <= CoScheduler.InputPrograms; i++)
        //    {
        //        for (int j = 1; j <= CoScheduler.InputPrograms; j++)
        //        {
        //            if (i != j)
        //            {
        //                P1 = DB./*PSTables*/.Where(P => P.ID == i).Select(P => P).Single();
        //                P2 = DB./*PSTables*/.Where(P => P.ID == j).Select(P => P).Single();
        //                if (P1.Kernel >= P2.MemcpyH2D)
        //                {
        //                    if (P1.MemcpyD2H >= P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //                else
        //                {
        //                    if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}

        //private void XIVStreamSetGenerator()
        //{
        //    LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
        //    for (int i = 0; i < 1000; i++)
        //    {
        //        //DB.(i);
        //    }
        //    var P1 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    var P2 = DB./*PSTables*/.Where(P => P.ID == 1).Select(P => P).Single();
        //    double H2D, Kernel, D2H;
        //    for (int i = 1; i <= CoScheduler.InputPrograms; i++)
        //    {
        //        for (int j = 1; j <= CoScheduler.InputPrograms; j++)
        //        {
        //            if (i != j)
        //            {
        //                P1 = DB./*PSTables*/.Where(P => P.ID == i).Select(P => P).Single();
        //                P2 = DB./*PSTables*/.Where(P => P.ID == j).Select(P => P).Single();
        //                if (P1.Kernel >= P2.MemcpyH2D)
        //                {
        //                    if (P1.MemcpyD2H >= P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = (P1.Kernel - P2.MemcpyH2D) + P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //                else
        //                {
        //                    if (P1.Kernel + P1.MemcpyD2H >= P2.MemcpyH2D + P2.Kernel)
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = ((P1.Kernel + P1.MemcpyD2H) - (P2.MemcpyH2D + P2.Kernel)) + P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                    else
        //                    {
        //                        H2D = P1.MemcpyH2D + P2.MemcpyH2D;
        //                        Kernel = P2.Kernel;
        //                        D2H = P2.MemcpyD2H;
        //                        //DB.(H2D, Kernel, D2H, P1.ID, P2.ID);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}

    }
}
