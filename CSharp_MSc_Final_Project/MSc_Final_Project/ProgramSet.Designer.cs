namespace MSc_Final_Project
{
    partial class ProgramSet
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
            this.PS_GridView = new System.Windows.Forms.DataGridView();
            this.PSFile_Btn = new System.Windows.Forms.Button();
            this.PSFileLocation_TxtBox = new System.Windows.Forms.TextBox();
            this.PSWorksheet_ComboBox = new System.Windows.Forms.ComboBox();
            this.PSFileWorksheet_Lbl = new System.Windows.Forms.Label();
            this.PSFileLoad_Btn = new System.Windows.Forms.Button();
            this.PSopenFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.SavePS_Btn = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.PS_GridView)).BeginInit();
            this.SuspendLayout();
            // 
            // PS_GridView
            // 
            this.PS_GridView.AllowUserToAddRows = false;
            this.PS_GridView.AllowUserToDeleteRows = false;
            this.PS_GridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.PS_GridView.Location = new System.Drawing.Point(14, 132);
            this.PS_GridView.Name = "PS_GridView";
            this.PS_GridView.ReadOnly = true;
            this.PS_GridView.RowHeadersVisible = false;
            this.PS_GridView.Size = new System.Drawing.Size(924, 320);
            this.PS_GridView.TabIndex = 0;
            // 
            // PSFile_Btn
            // 
            this.PSFile_Btn.Location = new System.Drawing.Point(68, 34);
            this.PSFile_Btn.Name = "PSFile_Btn";
            this.PSFile_Btn.Size = new System.Drawing.Size(118, 25);
            this.PSFile_Btn.TabIndex = 1;
            this.PSFile_Btn.Text = "ProgramSet File";
            this.PSFile_Btn.UseVisualStyleBackColor = true;
            this.PSFile_Btn.Click += new System.EventHandler(this.PSFile_Btn_Click);
            // 
            // PSFileLocation_TxtBox
            // 
            this.PSFileLocation_TxtBox.Location = new System.Drawing.Point(222, 37);
            this.PSFileLocation_TxtBox.Name = "PSFileLocation_TxtBox";
            this.PSFileLocation_TxtBox.ReadOnly = true;
            this.PSFileLocation_TxtBox.Size = new System.Drawing.Size(333, 21);
            this.PSFileLocation_TxtBox.TabIndex = 2;
            // 
            // PSWorksheet_ComboBox
            // 
            this.PSWorksheet_ComboBox.FormattingEnabled = true;
            this.PSWorksheet_ComboBox.Location = new System.Drawing.Point(222, 79);
            this.PSWorksheet_ComboBox.Name = "PSWorksheet_ComboBox";
            this.PSWorksheet_ComboBox.Size = new System.Drawing.Size(140, 22);
            this.PSWorksheet_ComboBox.TabIndex = 3;
            this.PSWorksheet_ComboBox.SelectedIndexChanged += new System.EventHandler(this.PSWorksheet_ComboBox_SelectedIndexChanged);
            // 
            // PSFileWorksheet_Lbl
            // 
            this.PSFileWorksheet_Lbl.AutoSize = true;
            this.PSFileWorksheet_Lbl.Location = new System.Drawing.Point(121, 82);
            this.PSFileWorksheet_Lbl.Name = "PSFileWorksheet_Lbl";
            this.PSFileWorksheet_Lbl.Size = new System.Drawing.Size(65, 14);
            this.PSFileWorksheet_Lbl.TabIndex = 4;
            this.PSFileWorksheet_Lbl.Text = "Worksheet: ";
            // 
            // PSFileLoad_Btn
            // 
            this.PSFileLoad_Btn.Location = new System.Drawing.Point(604, 34);
            this.PSFileLoad_Btn.Name = "PSFileLoad_Btn";
            this.PSFileLoad_Btn.Size = new System.Drawing.Size(121, 67);
            this.PSFileLoad_Btn.TabIndex = 5;
            this.PSFileLoad_Btn.Text = "Load";
            this.PSFileLoad_Btn.UseVisualStyleBackColor = true;
            this.PSFileLoad_Btn.Click += new System.EventHandler(this.PSFileLoad_Btn_Click);
            // 
            // PSopenFileDialog
            // 
            this.PSopenFileDialog.FileName = "openFileDialog";
            // 
            // SavePS_Btn
            // 
            this.SavePS_Btn.Location = new System.Drawing.Point(758, 34);
            this.SavePS_Btn.Name = "SavePS_Btn";
            this.SavePS_Btn.Size = new System.Drawing.Size(121, 67);
            this.SavePS_Btn.TabIndex = 6;
            this.SavePS_Btn.Text = "Save into DataBase";
            this.SavePS_Btn.UseVisualStyleBackColor = true;
            this.SavePS_Btn.Click += new System.EventHandler(this.SavePS_Btn_Click);
            // 
            // ProgramSet
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 14F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(950, 465);
            this.Controls.Add(this.SavePS_Btn);
            this.Controls.Add(this.PSFileLoad_Btn);
            this.Controls.Add(this.PSFileWorksheet_Lbl);
            this.Controls.Add(this.PSWorksheet_ComboBox);
            this.Controls.Add(this.PSFileLocation_TxtBox);
            this.Controls.Add(this.PSFile_Btn);
            this.Controls.Add(this.PS_GridView);
            this.Font = new System.Drawing.Font("Linux Biolinum", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.MaximizeBox = false;
            this.MaximumSize = new System.Drawing.Size(966, 504);
            this.MinimumSize = new System.Drawing.Size(966, 504);
            this.Name = "ProgramSet";
            this.Text = "ProgramSet";
            this.Load += new System.EventHandler(this.ProgramSet_Load);
            ((System.ComponentModel.ISupportInitialize)(this.PS_GridView)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.DataGridView PS_GridView;
        private System.Windows.Forms.Button PSFile_Btn;
        private System.Windows.Forms.TextBox PSFileLocation_TxtBox;
        private System.Windows.Forms.ComboBox PSWorksheet_ComboBox;
        private System.Windows.Forms.Label PSFileWorksheet_Lbl;
        private System.Windows.Forms.Button PSFileLoad_Btn;
        private System.Windows.Forms.OpenFileDialog PSopenFileDialog;
        private System.Windows.Forms.Button SavePS_Btn;
    }
}