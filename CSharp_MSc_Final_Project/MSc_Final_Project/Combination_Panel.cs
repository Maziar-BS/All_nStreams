using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MSc_Final_Project
{
    public partial class Combination_Panel : Form
    {
        public Combination_Panel()
        {
            InitializeComponent();
        }

        private void Combination_Panel_Load(object sender, EventArgs e)
        {
            Comb_GridView.ColumnCount = CoScheduler.NumofCombs;
            for (int i = 0; i < CoScheduler.NumofCombs; i++)
            {
                Comb_GridView.Columns[i].Name = "C" + i.ToString();
            }
            Comb_GridView.Rows.Add();
        }

        private void SaveCombs_Btn_Click(object sender, EventArgs e)
        {
            LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
            switch (CoScheduler.NumofCombs)
            {
                case 2:
                    DB.SaveCombinationTable(Comb_GridView.Rows.Count, Convert.ToInt32(Comb_GridView.Rows[0].Cells[0].Value), Convert.ToInt32(Comb_GridView.Rows[0].Cells[1].Value), 0, 0);
                    break;
                case 3:
                    DB.SaveCombinationTable(Comb_GridView.Rows.Count, Convert.ToInt32(Comb_GridView.Rows[0].Cells[0].Value), Convert.ToInt32(Comb_GridView.Rows[0].Cells[1].Value), Convert.ToInt32(Comb_GridView.Rows[0].Cells[2].Value), 0);
                    break;
                case 4:
                    DB.SaveCombinationTable(Comb_GridView.Rows.Count, Convert.ToInt32(Comb_GridView.Rows[0].Cells[0].Value), Convert.ToInt32(Comb_GridView.Rows[0].Cells[1].Value), Convert.ToInt32(Comb_GridView.Rows[0].Cells[2].Value), Convert.ToInt32(Comb_GridView.Rows[0].Cells[3].Value));
                    break;
            }
        }

        private void RemCombs_Btn_Click(object sender, EventArgs e)
        {
            LinqDataContext DB = new LinqDataContext(CoScheduler.ConnectionString);
            DB.DeleteCombinationsTable(Comb_GridView.Rows.Count);
        }
    }
}
