namespace MSc_Final_Project
{
    partial class CoScheduler
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(CoScheduler));
            this.toolStrip = new System.Windows.Forms.ToolStrip();
            this.Combs_Btn = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator5 = new System.Windows.Forms.ToolStripSeparator();
            this.ProgramSet_Btn = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.ActualExe_Btn = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.StreamSetPrediction_Btn = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
            this.SchedulerPrecision_Btn = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator4 = new System.Windows.Forms.ToolStripSeparator();
            this.PredictMemoryCopies_Btn = new System.Windows.Forms.ToolStripButton();
            this.openFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.NumofInputPrograms_TxtBot = new System.Windows.Forms.TextBox();
            this.InputPrograms_Lbl = new System.Windows.Forms.Label();
            this.NumofCombs_Lbl = new System.Windows.Forms.Label();
            this.H2DAsynccpyBW_Lbl = new System.Windows.Forms.Label();
            this.NumofCombs_ComboBox = new System.Windows.Forms.ComboBox();
            this.H2DAsynccpyBW_TxtBox = new System.Windows.Forms.TextBox();
            this.H2DSynccpyBW_Lbl = new System.Windows.Forms.Label();
            this.H2DSynccpyBW_TxtBox = new System.Windows.Forms.TextBox();
            this.ResetSetting_Btn = new System.Windows.Forms.Button();
            this.UpdateSetting_Btn = new System.Windows.Forms.Button();
            this.D2HAsynccpyBW_TxtBox = new System.Windows.Forms.TextBox();
            this.D2HSynccpyBW_TxtBox = new System.Windows.Forms.TextBox();
            this.D2HAsynccpyBW_Lbl = new System.Windows.Forms.Label();
            this.D2HSynccpyBW_Lbl = new System.Windows.Forms.Label();
            this.toolStrip.SuspendLayout();
            this.SuspendLayout();
            // 
            // toolStrip
            // 
            this.toolStrip.Font = new System.Drawing.Font("Linux Biolinum", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.toolStrip.GripStyle = System.Windows.Forms.ToolStripGripStyle.Hidden;
            this.toolStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.Combs_Btn,
            this.toolStripSeparator5,
            this.ProgramSet_Btn,
            this.toolStripSeparator1,
            this.ActualExe_Btn,
            this.toolStripSeparator2,
            this.StreamSetPrediction_Btn,
            this.toolStripSeparator3,
            this.SchedulerPrecision_Btn,
            this.toolStripSeparator4,
            this.PredictMemoryCopies_Btn});
            this.toolStrip.Location = new System.Drawing.Point(0, 0);
            this.toolStrip.Name = "toolStrip";
            this.toolStrip.Size = new System.Drawing.Size(694, 25);
            this.toolStrip.TabIndex = 0;
            this.toolStrip.Text = "toolStrip";
            // 
            // Combs_Btn
            // 
            this.Combs_Btn.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.Combs_Btn.Image = ((System.Drawing.Image)(resources.GetObject("Combs_Btn.Image")));
            this.Combs_Btn.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.Combs_Btn.Name = "Combs_Btn";
            this.Combs_Btn.Size = new System.Drawing.Size(84, 22);
            this.Combs_Btn.Text = "Combinations";
            this.Combs_Btn.Click += new System.EventHandler(this.Combs_Btn_Click);
            // 
            // toolStripSeparator5
            // 
            this.toolStripSeparator5.Name = "toolStripSeparator5";
            this.toolStripSeparator5.Size = new System.Drawing.Size(6, 25);
            // 
            // ProgramSet_Btn
            // 
            this.ProgramSet_Btn.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.ProgramSet_Btn.Image = ((System.Drawing.Image)(resources.GetObject("ProgramSet_Btn.Image")));
            this.ProgramSet_Btn.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.ProgramSet_Btn.Name = "ProgramSet_Btn";
            this.ProgramSet_Btn.Size = new System.Drawing.Size(76, 22);
            this.ProgramSet_Btn.Text = "Program Set";
            this.ProgramSet_Btn.Click += new System.EventHandler(this.ProgramSet_Btn_Click);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 25);
            // 
            // ActualExe_Btn
            // 
            this.ActualExe_Btn.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.ActualExe_Btn.Image = ((System.Drawing.Image)(resources.GetObject("ActualExe_Btn.Image")));
            this.ActualExe_Btn.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.ActualExe_Btn.Name = "ActualExe_Btn";
            this.ActualExe_Btn.Size = new System.Drawing.Size(105, 22);
            this.ActualExe_Btn.Text = "Actual Executions";
            this.ActualExe_Btn.Click += new System.EventHandler(this.ActualExe_Btn_Click);
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(6, 25);
            // 
            // StreamSetPrediction_Btn
            // 
            this.StreamSetPrediction_Btn.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.StreamSetPrediction_Btn.Image = ((System.Drawing.Image)(resources.GetObject("StreamSetPrediction_Btn.Image")));
            this.StreamSetPrediction_Btn.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.StreamSetPrediction_Btn.Name = "StreamSetPrediction_Btn";
            this.StreamSetPrediction_Btn.Size = new System.Drawing.Size(127, 22);
            this.StreamSetPrediction_Btn.Text = "StreamPair Prediction";
            this.StreamSetPrediction_Btn.Click += new System.EventHandler(this.ExeSets_Btn_Click);
            // 
            // toolStripSeparator3
            // 
            this.toolStripSeparator3.Name = "toolStripSeparator3";
            this.toolStripSeparator3.Size = new System.Drawing.Size(6, 25);
            // 
            // SchedulerPrecision_Btn
            // 
            this.SchedulerPrecision_Btn.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.SchedulerPrecision_Btn.Image = ((System.Drawing.Image)(resources.GetObject("SchedulerPrecision_Btn.Image")));
            this.SchedulerPrecision_Btn.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.SchedulerPrecision_Btn.Name = "SchedulerPrecision_Btn";
            this.SchedulerPrecision_Btn.Size = new System.Drawing.Size(123, 22);
            this.SchedulerPrecision_Btn.Text = "Optimum Stream Set";
            this.SchedulerPrecision_Btn.Click += new System.EventHandler(this.SchedulerPrecision_Btn_Click);
            // 
            // toolStripSeparator4
            // 
            this.toolStripSeparator4.Name = "toolStripSeparator4";
            this.toolStripSeparator4.Size = new System.Drawing.Size(6, 25);
            // 
            // PredictMemoryCopies_Btn
            // 
            this.PredictMemoryCopies_Btn.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.PredictMemoryCopies_Btn.Image = ((System.Drawing.Image)(resources.GetObject("PredictMemoryCopies_Btn.Image")));
            this.PredictMemoryCopies_Btn.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.PredictMemoryCopies_Btn.Name = "PredictMemoryCopies_Btn";
            this.PredictMemoryCopies_Btn.Size = new System.Drawing.Size(137, 22);
            this.PredictMemoryCopies_Btn.Text = "Predict Memory Copies";
            this.PredictMemoryCopies_Btn.Click += new System.EventHandler(this.PredictMemoryCopies_Btn_Click);
            // 
            // openFileDialog
            // 
            this.openFileDialog.FileName = "openFileDialog";
            // 
            // NumofInputPrograms_TxtBot
            // 
            this.NumofInputPrograms_TxtBot.Location = new System.Drawing.Point(197, 62);
            this.NumofInputPrograms_TxtBot.Name = "NumofInputPrograms_TxtBot";
            this.NumofInputPrograms_TxtBot.Size = new System.Drawing.Size(42, 20);
            this.NumofInputPrograms_TxtBot.TabIndex = 1;
            // 
            // InputPrograms_Lbl
            // 
            this.InputPrograms_Lbl.AutoSize = true;
            this.InputPrograms_Lbl.Location = new System.Drawing.Point(85, 65);
            this.InputPrograms_Lbl.Name = "InputPrograms_Lbl";
            this.InputPrograms_Lbl.Size = new System.Drawing.Size(108, 13);
            this.InputPrograms_Lbl.TabIndex = 2;
            this.InputPrograms_Lbl.Text = "Number of Programs: ";
            // 
            // NumofCombs_Lbl
            // 
            this.NumofCombs_Lbl.AutoSize = true;
            this.NumofCombs_Lbl.Location = new System.Drawing.Point(320, 62);
            this.NumofCombs_Lbl.Name = "NumofCombs_Lbl";
            this.NumofCombs_Lbl.Size = new System.Drawing.Size(128, 13);
            this.NumofCombs_Lbl.TabIndex = 3;
            this.NumofCombs_Lbl.Text = "Number of combinations: ";
            // 
            // H2DAsynccpyBW_Lbl
            // 
            this.H2DAsynccpyBW_Lbl.AutoSize = true;
            this.H2DAsynccpyBW_Lbl.Location = new System.Drawing.Point(80, 113);
            this.H2DAsynccpyBW_Lbl.Name = "H2DAsynccpyBW_Lbl";
            this.H2DAsynccpyBW_Lbl.Size = new System.Drawing.Size(183, 13);
            this.H2DAsynccpyBW_Lbl.TabIndex = 4;
            this.H2DAsynccpyBW_Lbl.Text = "Asyncronous HtoD Copy Bandwidth: ";
            // 
            // NumofCombs_ComboBox
            // 
            this.NumofCombs_ComboBox.FormattingEnabled = true;
            this.NumofCombs_ComboBox.Location = new System.Drawing.Point(464, 59);
            this.NumofCombs_ComboBox.Name = "NumofCombs_ComboBox";
            this.NumofCombs_ComboBox.Size = new System.Drawing.Size(42, 21);
            this.NumofCombs_ComboBox.TabIndex = 5;
            // 
            // H2DAsynccpyBW_TxtBox
            // 
            this.H2DAsynccpyBW_TxtBox.Location = new System.Drawing.Point(292, 110);
            this.H2DAsynccpyBW_TxtBox.Name = "H2DAsynccpyBW_TxtBox";
            this.H2DAsynccpyBW_TxtBox.Size = new System.Drawing.Size(223, 20);
            this.H2DAsynccpyBW_TxtBox.TabIndex = 6;
            // 
            // H2DSynccpyBW_Lbl
            // 
            this.H2DSynccpyBW_Lbl.AutoSize = true;
            this.H2DSynccpyBW_Lbl.Location = new System.Drawing.Point(85, 188);
            this.H2DSynccpyBW_Lbl.Name = "H2DSynccpyBW_Lbl";
            this.H2DSynccpyBW_Lbl.Size = new System.Drawing.Size(178, 13);
            this.H2DSynccpyBW_Lbl.TabIndex = 7;
            this.H2DSynccpyBW_Lbl.Text = "Syncronous HtoD Copy Bandwidth: ";
            // 
            // H2DSynccpyBW_TxtBox
            // 
            this.H2DSynccpyBW_TxtBox.Location = new System.Drawing.Point(292, 185);
            this.H2DSynccpyBW_TxtBox.Name = "H2DSynccpyBW_TxtBox";
            this.H2DSynccpyBW_TxtBox.Size = new System.Drawing.Size(223, 20);
            this.H2DSynccpyBW_TxtBox.TabIndex = 8;
            // 
            // ResetSetting_Btn
            // 
            this.ResetSetting_Btn.Location = new System.Drawing.Point(132, 274);
            this.ResetSetting_Btn.Name = "ResetSetting_Btn";
            this.ResetSetting_Btn.Size = new System.Drawing.Size(85, 37);
            this.ResetSetting_Btn.TabIndex = 9;
            this.ResetSetting_Btn.Text = "Reset";
            this.ResetSetting_Btn.UseVisualStyleBackColor = true;
            this.ResetSetting_Btn.Click += new System.EventHandler(this.ResetSetting_Btn_Click);
            // 
            // UpdateSetting_Btn
            // 
            this.UpdateSetting_Btn.Location = new System.Drawing.Point(397, 274);
            this.UpdateSetting_Btn.Name = "UpdateSetting_Btn";
            this.UpdateSetting_Btn.Size = new System.Drawing.Size(85, 37);
            this.UpdateSetting_Btn.TabIndex = 10;
            this.UpdateSetting_Btn.Text = "Update";
            this.UpdateSetting_Btn.UseVisualStyleBackColor = true;
            this.UpdateSetting_Btn.Click += new System.EventHandler(this.UpdateSetting_Btn_Click);
            // 
            // D2HAsynccpyBW_TxtBox
            // 
            this.D2HAsynccpyBW_TxtBox.Location = new System.Drawing.Point(292, 148);
            this.D2HAsynccpyBW_TxtBox.Name = "D2HAsynccpyBW_TxtBox";
            this.D2HAsynccpyBW_TxtBox.Size = new System.Drawing.Size(223, 20);
            this.D2HAsynccpyBW_TxtBox.TabIndex = 11;
            // 
            // D2HSynccpyBW_TxtBox
            // 
            this.D2HSynccpyBW_TxtBox.Location = new System.Drawing.Point(292, 225);
            this.D2HSynccpyBW_TxtBox.Name = "D2HSynccpyBW_TxtBox";
            this.D2HSynccpyBW_TxtBox.Size = new System.Drawing.Size(223, 20);
            this.D2HSynccpyBW_TxtBox.TabIndex = 12;
            // 
            // D2HAsynccpyBW_Lbl
            // 
            this.D2HAsynccpyBW_Lbl.AutoSize = true;
            this.D2HAsynccpyBW_Lbl.Location = new System.Drawing.Point(80, 151);
            this.D2HAsynccpyBW_Lbl.Name = "D2HAsynccpyBW_Lbl";
            this.D2HAsynccpyBW_Lbl.Size = new System.Drawing.Size(183, 13);
            this.D2HAsynccpyBW_Lbl.TabIndex = 13;
            this.D2HAsynccpyBW_Lbl.Text = "Asyncronous DtoH Copy Bandwidth: ";
            // 
            // D2HSynccpyBW_Lbl
            // 
            this.D2HSynccpyBW_Lbl.AutoSize = true;
            this.D2HSynccpyBW_Lbl.Location = new System.Drawing.Point(85, 228);
            this.D2HSynccpyBW_Lbl.Name = "D2HSynccpyBW_Lbl";
            this.D2HSynccpyBW_Lbl.Size = new System.Drawing.Size(178, 13);
            this.D2HSynccpyBW_Lbl.TabIndex = 14;
            this.D2HSynccpyBW_Lbl.Text = "Syncronous DtoH Copy Bandwidth: ";
            // 
            // CoScheduler
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(694, 341);
            this.Controls.Add(this.D2HSynccpyBW_Lbl);
            this.Controls.Add(this.D2HAsynccpyBW_Lbl);
            this.Controls.Add(this.D2HSynccpyBW_TxtBox);
            this.Controls.Add(this.D2HAsynccpyBW_TxtBox);
            this.Controls.Add(this.UpdateSetting_Btn);
            this.Controls.Add(this.ResetSetting_Btn);
            this.Controls.Add(this.H2DSynccpyBW_TxtBox);
            this.Controls.Add(this.H2DSynccpyBW_Lbl);
            this.Controls.Add(this.H2DAsynccpyBW_TxtBox);
            this.Controls.Add(this.NumofCombs_ComboBox);
            this.Controls.Add(this.H2DAsynccpyBW_Lbl);
            this.Controls.Add(this.NumofCombs_Lbl);
            this.Controls.Add(this.InputPrograms_Lbl);
            this.Controls.Add(this.NumofInputPrograms_TxtBot);
            this.Controls.Add(this.toolStrip);
            this.Font = new System.Drawing.Font("Linux Biolinum", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.MaximizeBox = false;
            this.MaximumSize = new System.Drawing.Size(710, 380);
            this.MinimumSize = new System.Drawing.Size(710, 380);
            this.Name = "CoScheduler";
            this.Text = "CoScheduler";
            this.Load += new System.EventHandler(this.CoScheduler_Load);
            this.toolStrip.ResumeLayout(false);
            this.toolStrip.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ToolStrip toolStrip;
        private System.Windows.Forms.ToolStripButton ProgramSet_Btn;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripButton ActualExe_Btn;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
        private System.Windows.Forms.ToolStripButton SchedulerPrecision_Btn;
        private System.Windows.Forms.OpenFileDialog openFileDialog;
        private System.Windows.Forms.ToolStripButton StreamSetPrediction_Btn;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;
        private System.Windows.Forms.TextBox NumofInputPrograms_TxtBot;
        private System.Windows.Forms.Label InputPrograms_Lbl;
        private System.Windows.Forms.Label NumofCombs_Lbl;
        private System.Windows.Forms.Label H2DAsynccpyBW_Lbl;
        private System.Windows.Forms.ComboBox NumofCombs_ComboBox;
        private System.Windows.Forms.TextBox H2DAsynccpyBW_TxtBox;
        private System.Windows.Forms.Label H2DSynccpyBW_Lbl;
        private System.Windows.Forms.TextBox H2DSynccpyBW_TxtBox;
        private System.Windows.Forms.Button ResetSetting_Btn;
        private System.Windows.Forms.Button UpdateSetting_Btn;
        private System.Windows.Forms.TextBox D2HAsynccpyBW_TxtBox;
        private System.Windows.Forms.TextBox D2HSynccpyBW_TxtBox;
        private System.Windows.Forms.Label D2HAsynccpyBW_Lbl;
        private System.Windows.Forms.Label D2HSynccpyBW_Lbl;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator4;
        private System.Windows.Forms.ToolStripButton PredictMemoryCopies_Btn;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator5;
        private System.Windows.Forms.ToolStripButton Combs_Btn;
    }
}

