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
using System.IO;

namespace MSc_Final_Project
{
    public partial class CoScheduler : Form
    {
        public static String ConnectionString = @"Data Source = (LocalDB)\MSSQLLocalDB;AttachDbFilename=" + Path.GetFullPath(@"..\..\..\") + @"SchedulerDataBase.mdf;Integrated Security=True;Connect Timeout=30";
        public static int InputPrograms, NumofCombs, NumofStreams;
        public static double H2DAsynccpyBW, D2HAsynccpyBW, H2DSynccpyBW, D2HSynccpyBW;
        public CoScheduler()
        {
            InitializeComponent();
        }

        private void CoScheduler_Load(object sender, EventArgs e)
        {
            for (int i = 1; i <= 32; i++)
            {
                NumofCombs_ComboBox.Items.Add(i);
            }
            InputPrograms = 14;
            NumofCombs = 2;
            NumofStreams = 3;
            H2DAsynccpyBW = 11500000000;
            D2HAsynccpyBW = 11500000000;
            H2DSynccpyBW = 4000000000;
            D2HSynccpyBW = 4000000000;
            NumofInputPrograms_TxtBot.Text = InputPrograms.ToString();
            H2DAsynccpyBW_TxtBox.Text = H2DAsynccpyBW.ToString();
            D2HAsynccpyBW_TxtBox.Text = D2HAsynccpyBW.ToString();
            H2DSynccpyBW_TxtBox.Text = H2DSynccpyBW.ToString();
            D2HSynccpyBW_TxtBox.Text = D2HSynccpyBW.ToString();
            NumofCombs_ComboBox.Text = "2";
        }

        private void ResetSetting_Btn_Click(object sender, EventArgs e)
        {
            InputPrograms = 14;
            NumofCombs = 2;
            NumofStreams = 3;
            H2DAsynccpyBW = 11500000000;
            D2HAsynccpyBW = 11500000000;
            H2DSynccpyBW = 4000000000;
            D2HSynccpyBW = 4000000000;
            NumofInputPrograms_TxtBot.Text = InputPrograms.ToString();
            H2DAsynccpyBW_TxtBox.Text = H2DAsynccpyBW.ToString();
            D2HAsynccpyBW_TxtBox.Text = D2HAsynccpyBW.ToString();
            H2DSynccpyBW_TxtBox.Text = H2DSynccpyBW.ToString();
            D2HSynccpyBW_TxtBox.Text = D2HSynccpyBW.ToString();
            NumofCombs_ComboBox.Text = "2";
        }

        private void UpdateSetting_Btn_Click(object sender, EventArgs e)
        {
            DialogResult DR = MessageBox.Show("Want to update the setting?", "Warning", MessageBoxButtons.YesNo, MessageBoxIcon.Warning);
            if (DR == DialogResult.Yes)
            {
                InputPrograms = Convert.ToInt32(NumofInputPrograms_TxtBot.Text);
                NumofCombs = Convert.ToInt32(NumofCombs_ComboBox.SelectedItem);
                H2DAsynccpyBW = Convert.ToDouble(H2DAsynccpyBW_TxtBox.Text);
                D2HAsynccpyBW = Convert.ToDouble(D2HAsynccpyBW_TxtBox.Text);
                H2DSynccpyBW = Convert.ToDouble(H2DSynccpyBW_TxtBox.Text);
                D2HSynccpyBW = Convert.ToDouble(D2HSynccpyBW_TxtBox.Text);
            }
        }

        private void ProgramSet_Btn_Click(object sender, EventArgs e)
        {
            ProgramSet PS = new ProgramSet();
            PS.ShowDialog();
        }

        private void Combs_Btn_Click(object sender, EventArgs e)
        {
            Combination_Panel CP = new Combination_Panel();
                CP.ShowDialog();
        }
        private void ActualExe_Btn_Click(object sender, EventArgs e)
        {
            ActualExecutions AE = new ActualExecutions();
            AE.ShowDialog();
        }

        private void ExeSets_Btn_Click(object sender, EventArgs e)
        {
            StreamSetPrediction ES = new StreamSetPrediction();
            ES.ShowDialog();
        }

        private void SchedulerPrecision_Btn_Click(object sender, EventArgs e)
        {
            SchedulerPrediction SP = new SchedulerPrediction();
            SP.ShowDialog();
        }
        private void PredictMemoryCopies_Btn_Click(object sender, EventArgs e)
        {
            MemCpyPrediction MCP = new MemCpyPrediction();
            MCP.ShowDialog();
        }

    }
}
